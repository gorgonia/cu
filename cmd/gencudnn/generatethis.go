package main

import (
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/cznic/cc"
	"github.com/gorgonia/bindgen"
	"github.com/kr/pretty"
)

// generate this contains function to generate for THIS package (main)

// generateMappings is used to generate the mappings
func generateMappings(appendCurrent bool) {
	hdr := "package main\n"

	initfn := `
	func init() {


	`
	t, err := bindgen.Parse(model, hdrfile)
	handleErr(err)

	var buf io.WriteCloser
	if appendCurrent {
		buf, err = os.OpenFile("mappings.go", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	} else {
		buf, err = os.OpenFile("mappings.go", os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	}
	handleErr(err)
	defer buf.Close()

	if !appendCurrent {
		fmt.Fprintln(buf, hdr)
		bindgen.GenIgnored(buf, t, functions)
		fmt.Fprintln(buf, initfn)
		bindgen.GenNameMap(buf, t, "fnNameMap", processNameBasic, functions, true)
		bindgen.GenNameMap(buf, t, "enumMappings", processNameBasic, enums, true)
		generateContextualNameMap(buf, t)
		generateAlphaBeta(buf, t)
		generateRetvals(buf, t)

		generateCRUD(buf, t, "create")
		generateCRUD(buf, t, "set")
		generateCRUD(buf, t, "destroy")
		generateCRUD(buf, t, "methods")
		fmt.Fprintln(buf, "}\n")
	}
	fmt.Fprintln(buf, initfn)

	fmt.Fprintln(buf, "}\n")
}

func generateContextualNameMap(buf io.Writer, t *cc.TranslationUnit) {
	filter := func(d *cc.Declarator) bool {
		if !functions(d) {
			return false
		}
		ps, _ := d.Type.Parameters()
		for _, p := range ps {
			if bindgen.NameOf(p.Declarator) == "handle" {
				return true
			}
		}
		return false
	}

	decls, err := bindgen.Get(t, filter)
	handleErr(err)
	fmt.Fprint(buf, "contextual = map[string]struct{} {\n")
	for _, d := range decls {
		fmt.Fprintf(buf, "%q: {},\n", d.(*bindgen.CSignature).Name)
	}
	fmt.Fprint(buf, "}\n")
}

func generateRetvals(buf io.Writer, t *cc.TranslationUnit) {
	decls, err := bindgen.Get(t, functions)
	handleErr(err)
	fmt.Fprint(buf, "retVals = map[string]map[int]string {\n")
	for _, d := range decls {
		cs := d.(*bindgen.CSignature)
		params := cs.Parameters()
		lname := strings.ToLower(cs.Name)

		if !strings.Contains(lname, "get") && !strings.Contains(lname, "create") {
			continue
		}

		for i := len(params) - 1; i >= 0; i-- {
			p := params[i]
			if !bindgen.IsConstType(p.Type()) && bindgen.IsPointer(p.Type()) {
				fmt.Fprintf(buf, "%q: {%d: %q},\n", cs.Name, i, p.Name())
				break
			}
		}
	}
	fmt.Fprint(buf, "}\n")
}

// generateCRUD creates lists of CRUD functions for the generateStubs function to consume
func generateCRUD(buf io.Writer, t *cc.TranslationUnit, fnType string) {
	decls, err := bindgen.Get(t, functions)
	handleErr(err)

	var a = make(map[string][]string)
	switch fnType {
	case "create":
		fmt.Fprintf(buf, "creations = ")
	case "set":
		fmt.Fprintf(buf, "setFns = ")
	case "destroy":
		fmt.Fprintf(buf, "destructions = ")
	case "methods":
		fmt.Fprintf(buf, "methods = ")
	}

	var b map[string]struct{}

	for _, d := range decls {
		cs := d.(*bindgen.CSignature)
		params := cs.Parameters()

		if fnType != "methods" && !strings.Contains(strings.ToLower(cs.Name), fnType) {
			continue
		}
		if fnType == "methods" && len(params) == 0 {
			continue
		}

		switch fnType {
		case "create":
			for i := len(params) - 1; i >= 0; i-- {
				p := params[i]
				if !bindgen.IsConstType(p.Type()) && bindgen.IsPointer(p.Type()) {
					typ := nameOfType(p.Type())
					a[typ] = append(a[typ], cs.Name)
				}
			}

		case "set", "destroy":
			p := params[0]
			typ := nameOfType(p.Type())
			if typ == "cudnnHandle_t" && len(params) > 1 {
				p = params[1]
				typ = nameOfType(p.Type())
			}
			a[typ] = append(a[typ], cs.Name)
		case "methods":
			if strings.Contains(strings.ToLower(cs.Name), "get") {
				continue
			}
			if _, ok := ignored[cs.Name]; ok {
				continue
			}
			if alreadyGenIn(cs.Name, creations, setFns, destructions) {
				continue
			}

			p := params[0]
			typ := nameOfType(p.Type())
			// if typ == "cudnnHandle_t" {
			// 	if len(params) == 1 {
			// 		continue
			// 	}
			// 	p = params[1]
			// 	typ = nameOfType(p.Type())
			// 	if alreadyDeclaredType(typ, enumMappings, manualChecks) {
			// 		if b == nil {
			// 			b = make(map[string]struct{})
			// 		}
			// 		b[cs.Name] = struct{}{}
			// 		continue
			// 	}
			// }
			a[typ] = append(a[typ], cs.Name)
		}
	}
	fmt.Fprintf(buf, "%# v\n\n", pretty.Formatter(a))

	// set the actual thing if not set
	// lisp users just shake their head in disappointment
	switch fnType {
	case "create":
		creations = a
	case "set":
		setFns = a
	case "destroy":
		destructions = a
	case "methods":
		methods = a
		if b != nil {
			fmt.Fprintf(buf, "orphaned = %# v\n\n", pretty.Formatter(b))
			orphaned = b
		}
	}
}

func generateAlphaBeta(buf io.Writer, t *cc.TranslationUnit) {
	decls, err := bindgen.Get(t, functions)
	handleErr(err)
	fmt.Fprint(buf, "alphaBetas = map[string]map[int]string {\n")
	for _, d := range decls {
		cs := d.(*bindgen.CSignature)
		params := cs.Parameters()

		var printedName bool
		for i := len(params) - 1; i >= 0; i-- {
			p := params[i]
			if !(bindgen.IsConstType(p.Type()) && bindgen.IsPointer(p.Type())) {
				continue
			}

			switch p.Name() {
			case "alpha", "alpha1", "alpha2", "alpha3", "beta", "beta1":
				if !printedName {
					printedName = true
					fmt.Fprintf(buf, "%q: {", cs.Name)
				}
				fmt.Fprintf(buf, "%d: %q, ", i, p.Name())
			default:
			}

		}
		if printedName {
			fmt.Fprint(buf, "},\n") // close the first
		}
	}
	fmt.Fprint(buf, "}\n")
}
