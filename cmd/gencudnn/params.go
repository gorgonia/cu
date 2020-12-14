package main

import (
	bg "github.com/gorgonia/bindgen"
	"modernc.org/cc"
)

func isInput(fnName string, p bg.Parameter) bool {
	return inList(p.Name(), inputParams[fnName])
}

func isOutput(fnName string, p bg.Parameter) bool {
	return inList(p.Name(), outputParams[fnName])
}

func isIO(fnName string, p bg.Parameter) bool {
	return inList(p.Name(), ioParams[fnName])
}

func isAlphaBeta(fnName string, p bg.Parameter) bool {
	locs := alphaBetas[fnName]
	for _, v := range locs {
		if v == p.Name() {
			return true
		}
	}
	return false
}

// functions for convertibility
func isOutputPtrOfPrim(fnName string, p bg.Parameter) bool {
	if !isOutput(fnName, p) && !isIO(fnName, p) {
		return false
	}
	if !p.IsPointer() {
		return false
	}
	return isBuiltin(depointerize(nameOfType(p.Type())))
}

func isEnumOutput(fnName string, p bg.Parameter) bool {
	if !isOutput(fnName, p) {
		return false
	}
	if !p.IsPointer() {
		return false
	}
	cType := nameOfType(p.Type())
	_, ok := enumMappings[cType]
	return ok
}

func cParam2GoParam(p bg.Parameter) (retVal Param) {
	retVal.Name = safeParamName(p.Name())
	cTypeName := nameOfType(p.Type())
	gTypeName := goNameOf(p.Type())
	isPtr, isBuiltin := isPointerOfBuiltin(cTypeName)

	switch {
	case gTypeName == "" && isPtr && isBuiltin:
		retVal.Type = goNameOfStr(depointerize(cTypeName))
	case gTypeName != "":
		retVal.Type = gTypeName
	case gTypeName == "" && !isBuiltin:
		retVal.Type = "TODO"
	}
	return
}

func ctype2gotype2ctype(t cc.Type) string {
	cName := nameOfType(t)
	goName := goNameOfStr(depointerize(cName))
	return go2cBuiltins[goName]
}
