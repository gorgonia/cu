package main

import (
	"bytes"
	"fmt"
	"log"
	"strings"
)

// Param represents a parameter in the signature
type Param struct {
	Name        string
	Type        string
	IsPtr       bool
	IsRetVal    bool // if it's a RetVal, the address is taken instead
	Convertible bool // there exists a conversion function from Go->C and vice versa
}

func NewParam(name, typ string, isPtr bool) *Param {
	return &Param{
		Name:  name,
		Type:  typ,
		IsPtr: isPtr,
	}
}

func (p *Param) String() string {
	return fmt.Sprintf("%q :: %v(%t,%t)", p.Name, p.Type, p.IsPtr, p.IsRetVal)
}

// CSignature represents a signature
type CSignature struct {
	Name   string
	Params []*Param
}

func (sig *CSignature) String() string {
	var buf bytes.Buffer
	buf.WriteString(sig.Name)
	buf.WriteString("\n")
	for _, param := range sig.Params {
		fmt.Fprintf(&buf, "\t%v\n", param)
	}
	return buf.String()
}

func (sig *CSignature) ParamByName(name string) *Param {
	for _, p := range sig.Params {
		if p.Name == name {
			return p
		}
	}
	return nil
}

func (sig *CSignature) IsGet() bool {
	if strings.Contains(sig.Name, "Get") || strings.Contains(sig.Name, "Create") {
		return true
	}

	for _, ret := range returns {
		if sig.Name == ret {
			return true
		}
	}
	return false
}

func (sig *CSignature) Fix() {
	if sig.IsGet() {
		for _, param := range sig.Params {
			if param.IsPtr {
				param.IsRetVal = true
			}
		}
	}
}

func (sig *CSignature) GoSig() *GoSignature {
	name, ok := fnNameMap[sig.Name]
	if !ok {
		err := fmt.Sprintf("Name %q not found in mapping", sig.Name)
		errs[err] = struct{}{}
	}

	name, receiver := splitReceiver(name)
	ignored := -1

	var receiverParam *Param
	if receiver != "" {
		receiverParam = new(Param)
		// search the C signature for the first input with the type
		for i, p := range sig.Params {
			if gt, ok := goTypeFromCtype(p.Type); ok && gt == receiver {
				ignored = i

				receiverParam.Type = gt
				receiverParam.Name = p.Name
				break
			}
		}
		if receiverParam.Name == "" {
			err := fmt.Sprintf("Receiver %q not found in signature of %v", receiver, sig.Name)
			errs[err] = struct{}{}
			return nil
		}
	}

	var params []*Param
	var retVals []*Param
	for i, p := range sig.Params {
		if i == ignored {
			continue
		}
		// void param means... no input
		if p.Type == "void" && !p.IsPtr {
			continue
		}

		gp := new(Param)
		switch {
		case strings.Contains(strings.ToLower(p.Name), "flag") && (strings.Contains(sig.Name, "Create") || strings.Contains(sig.Name, "SetFlag")):
			// if flag, the Go signature expects a typed flag
			gp.Type = flagType(sig.Name)
			gp.Name = p.Name
		default:
			var ok bool
			gp.Name = p.Name
			if gp.Type, ok = goTypeFromCtype(p.Type); !ok {
				err := fmt.Sprintf("ctype %q has no Go equivalent.", p.Type)
				errs[err] = struct{}{}
				continue
			}
		}

		if p.IsRetVal {
			retVals = append(retVals, gp)
		} else {
			params = append(params, gp)
		}
	}
	return &GoSignature{
		Name:     name,
		Receiver: receiverParam,
		Params:   params,
		RetVals:  retVals,
		CSig:     sig,
	}
}

// GoSignature represents a function signature in Go
type GoSignature struct {
	Name     string
	Receiver *Param
	Params   []*Param
	RetVals  []*Param
	CSig     *CSignature
}

func (sig *GoSignature) ParamByName(name string) *Param {
	for _, param := range sig.Params {
		if param.Name == name {
			return param
		}
	}

	for _, param := range sig.RetVals {
		if param.Name == name {
			return param
		}
	}
	return nil
}

func (sig *GoSignature) Format(f fmt.State, c rune) {
	f.Write([]byte("func "))
	if sig.Receiver != nil {
		fmt.Fprintf(f, "(%s %s) ", sig.Receiver.Name, sig.Receiver.Type)
	}
	fmt.Fprintf(f, "%s(", sig.Name)
	for i, p := range sig.Params {
		fmt.Fprintf(f, "%s %s", p.Name, p.Type)
		if i < len(sig.Params)-1 {
			fmt.Fprintf(f, ", ")
		}
	}
	f.Write([]byte(") "))
	if len(sig.RetVals) > 0 {
		f.Write([]byte("("))
		for i, retVal := range sig.RetVals {
			fmt.Fprintf(f, "%s %s", retVal.Name, retVal.Type)
			if i < len(sig.RetVals)-1 {
				fmt.Fprintf(f, ", ")
			}
		}
		f.Write([]byte(")"))
	}
}

func splitReceiver(s string) (name, receiver string) {
	if strings.Contains(s, " ") {
		splits := strings.Split(s, " ")
		name = splits[1]
		receiver = splits[0]
		return
	}
	return s, ""
}

// flag type returns the correct flag type for the Go signatures
func flagType(name string) string {
	switch name {
	case "cuCtxCreate", "cuDevicePrimaryCtxSetFlags":
		return "ContextFlags"
	case "cuLinkCreate":
		panic("Unhandled")
	case "cuStreamCreate", "cuStreamCreateWithPriority":
		return "StreamFlags"
	case "cuEventCreate":
		return "EventFlags"
	case "cuTexRefSetFlags":
		return "TexRefFlags"
	default:
		log.Printf("Unreachable flagtype %v", name)
	}
	// panic("Unreachable")
	return "UNKNOWN"
}

func goTypeFromCtype(ct string) (string, bool) {
	t, ok := ctypes2GoTypes["C."+ct]
	return t, ok
}
