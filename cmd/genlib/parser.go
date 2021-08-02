package main

import (
	"strings"

	"github.com/gorgonia/bindgen"
	"modernc.org/cc"
)

func Parse() (retVal []*CSignature) {
	t, err := bindgen.Parse(bindgen.Model(), "cuda.h")
	if err != nil {
		panic(err)
	}

	decls, err := functions(t)
	if err != nil {
		panic(err)
	}

	for _, d := range decls {
		retVal = append(retVal, decl2csig(d.(*bindgen.CSignature)))
	}
	return
}

func functions(t *cc.TranslationUnit) ([]bindgen.Declaration, error) {
	filter := func(decl *cc.Declarator) bool {
		name := bindgen.NameOf(decl)
		if !strings.HasPrefix(name, "cu") {
			return false
		}
		if _, ok := ignoredFunctions[name]; ok {
			return false
		}
		if decl.Type.Kind() == cc.Function {
			return true
		}
		return false
	}
	return bindgen.Get(t, filter)
}

func decl2csig(d *bindgen.CSignature) *CSignature {
	retVal := new(CSignature)
	retVal.Name = d.Name
	var params []*Param
	for _, p := range d.Parameters() {
		params = append(params, bgparam2param(p))
	}
	retVal.Params = params
	retVal.Fix()
	return retVal
}

// bgparam2cparam transforms bindgen parameter to *Param
func bgparam2param(p bindgen.Parameter) *Param {
	name := p.Name()
	typ := cleanType(p.Type())
	isPtr := bindgen.IsPointer(p.Type())
	return NewParam(name, typ, isPtr)
}

func cleanType(t cc.Type) string {
	typ := t.String()
	if td := bindgen.TypeDefOf(t); td != "" {
		typ = td
	}

	if bindgen.IsConstType(t) {
		typ = strings.TrimPrefix(typ, "const ")
	}

	if bindgen.IsPointer(t) {
		typ = strings.TrimSuffix(typ, "*")
	}
	return typ
}
