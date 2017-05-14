package main

import (
	"fmt"
	"io"
)

func generateAPI(buf io.Writer, sigs []*GoSignature) {
	for _, sig := range sigs {
		writeAPISig(buf, sig)
		fmt.Fprintf(buf, "{\n")
		writeFunctionBody(buf, sig)
		fmt.Fprintf(buf, "} \n\n")
	}
}

func generateContextAPI(buf io.Writer, sigs []*GoSignature) {
	for _, sig := range sigs {
		writeContextSig(buf, sig)
		fmt.Fprintf(buf, "{\n")
		writeContextMethodBody(buf, sig)
		fmt.Fprintf(buf, "}\n\n")
	}
}

func writeFunctionBody(buf io.Writer, sig *GoSignature) {
	csig := sig.CSig

	if sig.Receiver != nil {
		cParam := csig.ParamByName(sig.Receiver.Name)
		go2CParam(buf, sig.Receiver, cParam)
	}

	for _, param := range sig.Params {
		cParam := csig.ParamByName(param.Name)
		go2CParam(buf, param, cParam)
	}

	for _, ret := range sig.RetVals {
		cParam := csig.ParamByName(ret.Name)
		fmt.Fprintf(buf, "var C%s C.%s\n", cParam.Name, cParam.Type)
	}

	if len(sig.RetVals) == 0 {
		buf.Write([]byte("return "))
	} else {
		buf.Write([]byte("err = "))
	}

	cgoCall(buf, sig.CSig)

	if len(sig.RetVals) > 0 {
		for _, ret := range sig.RetVals {
			cParam := csig.ParamByName(ret.Name)
			var conv string

			if cf, ok := ctypesConversion["C."+cParam.Type]; ok {
				conv = fmt.Sprintf(cf, "C"+cParam.Name)
			} else {
				conv = fmt.Sprintf("%s(C%s)\n", ret.Type, cParam.Name)
			}

			fmt.Fprintf(buf, "%s = %s\n", ret.Name, conv)
		}
		buf.Write([]byte("return\n"))
	}
}

func writeContextMethodBody(buf io.Writer, sig *GoSignature) {
	csig := sig.CSig

	if sig.Receiver != nil {
		cParam := csig.ParamByName(sig.Receiver.Name)
		go2CParam(buf, sig.Receiver, cParam)
	}

	for _, param := range sig.Params {
		cParam := csig.ParamByName(param.Name)
		go2CParam(buf, param, cParam)
	}

	for _, ret := range sig.RetVals {
		cParam := csig.ParamByName(ret.Name)
		fmt.Fprintf(buf, "var C%s C.%s\n", cParam.Name, cParam.Type)
	}

	fmt.Fprintf(buf, "f := func() error { return ")
	cgoCall(buf, sig.CSig)
	fmt.Fprintf(buf, "}\n")

	if len(sig.RetVals) == 0 {
		buf.Write([]byte("ctx.err = ctx.Do(f) "))
	} else {
		fmt.Fprintf(buf, "if err = ctx.Do(f); err != nil {\n err = errors.Wrap(err, \"%s\")\n }\n", sig.Name)

	}

	if len(sig.RetVals) > 0 {
		for _, ret := range sig.RetVals {
			cParam := csig.ParamByName(ret.Name)
			var conv string

			if cf, ok := ctypesConversion["C."+cParam.Type]; ok {
				conv = fmt.Sprintf(cf, "C"+cParam.Name)
			} else {
				conv = fmt.Sprintf("%s(C%s)", ret.Type, cParam.Name)
			}

			fmt.Fprintf(buf, "%s = %s\n", ret.Name, conv)
		}
		buf.Write([]byte("return"))
	}
}

func writeAPISig(buf io.Writer, sig *GoSignature) {
	buf.Write([]byte("func "))
	if sig.Receiver != nil {
		fmt.Fprintf(buf, "(%s %s) ", sig.Receiver.Name, sig.Receiver.Type)
	}
	fmt.Fprintf(buf, "%s(", sig.Name)
	for i, p := range sig.Params {
		fmt.Fprintf(buf, "%s %s", p.Name, p.Type)
		if i < len(sig.Params)-1 {
			fmt.Fprintf(buf, ", ")
		}
	}
	buf.Write([]byte(") "))
	buf.Write([]byte("("))
	if len(sig.RetVals) > 0 {
		for i, retVal := range sig.RetVals {
			fmt.Fprintf(buf, "%s %s", retVal.Name, retVal.Type)
			if i < len(sig.RetVals) {
				fmt.Fprintf(buf, ", ")
			}
		}
	}

	buf.Write([]byte("err error)"))
}

func writeContextSig(buf io.Writer, sig *GoSignature) {
	buf.Write([]byte("func (ctx *Ctx) "))
	fmt.Fprintf(buf, "%s(", sig.Name)
	if sig.Receiver != nil {
		fmt.Fprintf(buf, "%s %s ", sig.Receiver.Name, sig.Receiver.Type)
		if len(sig.Params) > 0 {
			buf.Write([]byte(", "))
		}
	}

	for i, p := range sig.Params {
		fmt.Fprintf(buf, "%s %s", p.Name, p.Type)
		if i < len(sig.Params)-1 {
			fmt.Fprintf(buf, ", ")
		}
	}
	buf.Write([]byte(") "))
	buf.Write([]byte("("))
	if len(sig.RetVals) > 0 {
		for i, retVal := range sig.RetVals {
			fmt.Fprintf(buf, "%s %s", retVal.Name, retVal.Type)
			if i < len(sig.RetVals) {
				fmt.Fprintf(buf, ", ")
			}
		}
		buf.Write([]byte("err error"))
	}

	buf.Write([]byte(")"))
}
