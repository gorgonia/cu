package main

import (
	"fmt"
	"io"
)

func filterCSigs(sigs []*CSignature) (retVal []*CSignature) {
	for _, sig := range sigs {
		if _, ok := ignoredFunctions[sig.Name]; ok {
			continue
		}
		retVal = append(retVal, sig)
	}
	return
}

func cgoCall(buf io.Writer, sig *CSignature) {
	fmt.Fprintf(buf, "result(C.%s(", sig.Name)
	for i, param := range sig.Params {
		if param.Type == "void" && !param.IsPtr {
			continue
		}
		if param.IsPtr && param.IsRetVal {
			fmt.Fprintf(buf, "&C%s", param.Name)
		} else {
			fmt.Fprintf(buf, "C%s", param.Name)
		}

		if i < len(sig.Params)-1 {
			buf.Write([]byte(", "))
		}
	}
	fmt.Fprintf(buf, "))\n")
}

func go2CParam(buf io.Writer, goParam, cParam *Param) {
	if cParam == nil {
		panic("WTF?")
	}
	ctype, ok := gotypesConversion[goParam.Type]
	if !ok {
		panic(fmt.Sprintf("Go type %q does not have conversion to C type", goParam.Type))
	}
	conv := fmt.Sprintf(ctype, goParam.Name)
	fmt.Fprintf(buf, "C%s := %s\n", cParam.Name, conv)
}
