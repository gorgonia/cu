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

func parsePkg() *PkgState {
	filterFn := func(f os.FileInfo) bool {
		if strings.HasPrefix(f.Name(), "generated_") {
			return false
		}
		return true
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
