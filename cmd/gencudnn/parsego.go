package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"log"
	"os"
	"strings"
)

type PkgState struct {
	*ast.Package
}

func (s *PkgState) TypeDecls() []*ast.TypeSpec {
	var retVal []*ast.TypeSpec
	for _, f := range s.Files {
		for _, decl := range f.Decls {
			g, ok := decl.(*ast.GenDecl)
			if !ok {
				continue
			}
			for _, sp := range g.Specs {
				if ts, ok := sp.(*ast.TypeSpec); ok && ts.Name.Name != "TODO" {
					retVal = append(retVal, ts)
				}
			}
		}
	}
	return retVal
}

// checkNils checks the package for functions that return potentially nil pointer types.
//
// It expects functions to have return names. Which is what the generator generates anyways
func (s *PkgState) checkNils() []string {
	var retVal []string
	for _, f := range s.Files {
		for _, decl := range f.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok {
				continue
			}

			if fn.Type.Results == nil {
				continue
			}

			retVals := make(map[string]bool)  // name:isPtr
			retTypes := make(map[string]bool) // typeName: hasBeenAsigned
			posRetVal := make([]string, 0, len(fn.Type.Results.List))
			for _, ret := range fn.Type.Results.List {
				for _, name := range ret.Names {
					posRetVal = append(posRetVal, name.Name)
				}

				switch r := ret.Type.(type) {
				case *ast.StarExpr:
					if _, ok := r.X.(*ast.Ident); ok {
						for _, name := range ret.Names {
							retTypes[name.Name] = false
							retVals[name.Name] = true
						}
					}
				case *ast.Ident:
					// don't add to retTypes, but keep adding to retNames
					for _, name := range ret.Names {
						retVals[name.Name] = false
					}
				}
			}
			for _, stmt := range fn.Body.List {
				switch s := stmt.(type) {
				case *ast.AssignStmt:
					for _, lhs := range s.Lhs {
						if ident, ok := lhs.(*ast.Ident); ok {
							if _, ok := retTypes[ident.Name]; ok {
								retTypes[ident.Name] = true
							}
						}
					}
				case *ast.ReturnStmt:
					for i, ret := range s.Results {
						if ue, ok := ret.(*ast.UnaryExpr); ok && ue.Op == token.AND {
							retTypes[posRetVal[i]] = true // assume assigned
						}
					}
				}
			}

			for _, v := range retTypes {
				if !v {
					retVal = append(retVal, fn.Name.Name)
				}
			}
		}
	}
	return retVal
}

type usedCFnVisit struct {
	counter map[string]int
}

func (v *usedCFnVisit) Visit(node ast.Node) ast.Visitor {
	if cexpr, ok := node.(*ast.CallExpr); ok {
		if selExpr, ok := cexpr.Fun.(*ast.SelectorExpr); ok {
			if xid, ok := selExpr.X.(*ast.Ident); ok && xid.Name == "C" {
				v.counter[selExpr.Sel.Name]++
			}
		}
	}
	return v
}

// useCFn returns the count of how many times a C function has been used in the generated package
func (pkg *PkgState) usedCFn() map[string]int {
	retVal := make(map[string]int)
	visitor := &usedCFnVisit{retVal}
	for _, f := range pkg.Files {
		for _, decl := range f.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok {
				continue
			}
			ast.Walk(visitor, fn)
		}
	}
	return retVal
}

type usedCTypeVisit struct {
	counter map[string]int
}

func (v *usedCTypeVisit) Visit(node ast.Node) ast.Visitor {
	if ts, ok := node.(*ast.TypeSpec); ok {
		if st, ok := ts.Type.(*ast.StructType); ok {
			for _, field := range st.Fields.List {
				var hasInternal bool
				for _, name := range field.Names {
					if name.Name == "internal" {
						hasInternal = true
						break
					}
				}
				if !hasInternal {
					continue
				}
				if selExpr, ok := field.Type.(*ast.SelectorExpr); ok {
					if xid, ok := selExpr.X.(*ast.Ident); ok && xid.Name == "C" {
						v.counter[selExpr.Sel.Name]++
					}
				}
			}
		}
	}
	return v
}

func (pkg *PkgState) usedCTypes() map[string]int {
	retVal := make(map[string]int)
	visitor := &usedCTypeVisit{retVal}
	for _, f := range pkg.Files {
		for _, decl := range f.Decls {
			switch d := decl.(type) {
			case *ast.GenDecl:
				if d.Tok == token.TYPE {
					ast.Walk(visitor, d)
				}
			case *ast.FuncDecl:
				// handles enum conversions
				if d.Recv != nil && d.Name.Name == "C" {
					if selExpr, ok := d.Type.Results.List[0].Type.(*ast.SelectorExpr); ok {
						if xid, ok := selExpr.X.(*ast.Ident); ok && xid.Name == "C" {
							retVal[selExpr.Sel.Name]++
						}
					}
				}
			}
		}
	}

	return retVal
}

func parsePkg(ignoreGenerated bool) *PkgState {
	filterFn := func(f os.FileInfo) bool {
		if strings.HasPrefix(f.Name(), "generated_") {
			return false
		}
		return true
	}
	if ignoreGenerated {
		filterFn = nil
	}
	fs := token.NewFileSet()
	pkgs, err := parser.ParseDir(fs, pkgloc, filterFn, parser.AllErrors)
	if err != nil {
		log.Printf("Error parsing state: %v", err)
		return nil
	}
	if _, ok := pkgs["cudnn"]; !ok {
		log.Printf("Cannot find package cudnn")
		return nil
	}
	return &PkgState{pkgs["cudnn"]}
}

func alreadyProcessedType(typ string, decls []*ast.TypeSpec) bool {
	for _, d := range decls {
		if typ == d.Name.Name {
			return true
		}
	}
	return false
}
