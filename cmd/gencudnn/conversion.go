package main

import (
	"fmt"
	"log"
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

func csig2gosig(cs *bg.CSignature, retType string, returnsErr bool, retVal *GoSignature) (*GoSignature, error) {
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
		if _, ok := retValPos[i]; ok {
			continue
		}
		typeName := goNameOf(p.Type())
		if retVal.Receiver.Type == typeName {
			continue
		}
		if typeName == "" {
			err = errors.Errorf("%q: Parameter %d Skipped %q of %v - unmapped type", cs.Name, i, p.Name(), p.Type())
			log.Printf("%v (%d): param %v param type %v", cs.Name, i, p.Name(), p.Type())
			continue
		}
		paramName := p.Name()
		if inList(paramName, ioParamList) {
			retVal.Doc += fmt.Sprintf("\n//\t%v is both an input and output", paramName)
		}
		retVal.Params = append(retVal.Params, Param{Name: paramName, Type: reqPtr(typeName)})
	}

	var writeErrName bool
	if retType != "" {
		retVal.RetVals = append(retVal.RetVals, Param{Type: retType})
	} else {
		for pos := range retValPos {
			p := params[pos]
			typeName := goNameOf(p.Type())
			if typeName == "" {
				log.Printf("RetVal of %v (%d) cannot be generated: param %v param type %v", cs.Name, pos, p.Name(), p.Type())
				continue
			}
			retVal.RetVals = append(retVal.RetVals, Param{Name: p.Name(), Type: typeName})
			writeErrName = true
		}
	}

	switch {
	case returnsErr && !writeErrName:
		retVal.RetVals = append(retVal.RetVals, Param{Type: "error"})
	case returnsErr && writeErrName:
		retVal.RetVals = append(retVal.RetVals, Param{Name: "err", Type: "error"})
	}

	return retVal, err
}
