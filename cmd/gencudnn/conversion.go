package main

import (
	"fmt"
	"strings"

	bg "github.com/gorgonia/bindgen"
	"github.com/pkg/errors"
)

// Param represents a parameter in the signature
type Param struct {
	Name        string
	Type        string
	IsPtr       bool
	IsRetVal    bool // if it's a RetVal, the address is taken instead
	Convertible bool // there exists a conversion function from Go->C and vice versa
}

func MakeParam(name, typ string, isPtr bool) Param {
	return Param{
		Name:  name,
		Type:  typ,
		IsPtr: isPtr,
	}
}

func (p Param) String() string {
	if p.IsPtr {
		return fmt.Sprintf("%v *%v", p.Name, p.Type)
	}
	return fmt.Sprintf("%v %v", p.Name, p.Type)
}

// GoSignature represents a function signature in Go
type GoSignature struct {
	Name     string
	Receiver Param
	Params   []Param
	RetVals  []Param
	CSig     *bg.CSignature
	Doc      string
}

func (s GoSignature) Format(f fmt.State, c rune) {
	if s.Doc != "" {
		fmt.Fprintf(f, "// %s\n", s.Doc)
	}
	fmt.Fprint(f, "func ")
	if s.Receiver.Name != "" {
		fmt.Fprintf(f, "(%v) ", s.Receiver)
	}
	fmt.Fprintf(f, "%v(", s.Name)
	for _, p := range s.Params {
		fmt.Fprintf(f, "%v, ", p)
	}
	fmt.Fprint(f, ") ")
	if len(s.RetVals) > 0 {
		fmt.Fprint(f, "(")
		for _, r := range s.RetVals {
			fmt.Fprintf(f, "%v, ", r)
		}
		fmt.Fprint(f, ")")
	}
}

// HasAlphaBeta returns true if one of the parameters are alpha/beta
func (s GoSignature) AlphaBetas() []string {
	var retVal []string
	for _, p := range s.Params {
		if inList(p.Name, alphaBetaParams) {
			retVal = append(retVal, p.Name)
		}
	}
	return retVal
}

// FirstTensor finds the first tensor parameter of the function and returns the name
func (s GoSignature) FirstTensor() string {
	for _, p := range s.Params {
		if strings.Contains(p.Type, ctypes2GoTypes["cudnnTensorDescriptor_t"]) {
			return p.Name
		}
	}
	panic(fmt.Sprintf("No tensors to check in %v", s))
}

func csig2gosig(cs *bg.CSignature, retVal *GoSignature) (*GoSignature, error) {
	if retVal == nil {
		retVal = new(GoSignature)
	}
	if retVal.Name != "" {
		doc := docs[cs.Name]
		doc = strings.Replace(doc, cs.Name, retVal.Name, -1)
		doc = strings.Replace(doc, "\n", "\n// ", -1)
		retVal.Doc = doc
	}

	var err error
	params := cs.Parameters()
	retValPos := getRetValOnly(cs)

	ioParamList := ioParams[cs.Name]
	for i, p := range params {

		_, isRetVal := retValPos[i]
		name := p.Name()
		typeName := goNameOf(p.Type())

		// because the cuDNN library will not allocate on the users' behalf, any memory related stuff has to be preallocated by the user
		if isRetVal && typeName != "Memory" {
			continue
		}

		// all alpha/betas receive float64
		if isAlphaBeta(cs.Name, p) {
			goParam := cParam2GoParam(p)
			goParam.Type = "float64"
			retVal.Params = append(retVal.Params, goParam)
			continue
		}

		if retVal.Receiver.Type == typeName {
			continue
		}
		goParam := cParam2GoParam(p)
		if inList(name, ioParamList) {
			retVal.Doc += fmt.Sprintf("\n//\t%v is both an input and output", safeParamName(name))
		}
		retVal.Params = append(retVal.Params, goParam)
	}

	// handle retVals
	for i, p := range params {
		if !isOutput(cs.Name, p) {
			continue
		}
		goParam := cParam2GoParam(p)
		// we only ever need to return one error. It's enough for a //TODO notation
		if goParam.Type == "Memory" {
			err = errors.Errorf("%q returns Memory type in Parameter %d", cs.Name, i)
			continue
		}
		if retVal.Receiver.Type == goParam.Type && retVal.Receiver.Name == "DUMMY" {
			continue
		}
		retVal.RetVals = append(retVal.RetVals, goParam)
	}

	writeErrName := len(retVal.RetVals) > 0
	if writeErrName {
		retVal.RetVals = append(retVal.RetVals, Param{Name: "err", Type: "error"})
	} else {
		retVal.RetVals = append(retVal.RetVals, Param{Type: "error"})
	}

	return retVal, err
}
