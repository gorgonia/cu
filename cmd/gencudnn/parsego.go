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
					// don't add to retTypes
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
						if ue, ok := ret.(*ast.UnaryExpr); ok && ue.Op.String() == "&" {
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
