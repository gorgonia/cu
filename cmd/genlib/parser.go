package main

import (
	"io"
	"strings"
)

func Parse(input io.Reader) (retVal []*CSignature) {
	l := NewLexer("parse", input)
	go l.Run()

	var sig *CSignature
	for lex := range l.Output {
		switch lex.t {
		case RetType:
			if sig != nil {
				sig.Fix()
				retVal = append(retVal, sig)
			}
			sig = new(CSignature)
			continue
		case FunctionName:
			sig.Name = lex.v
		case Parameter:
			p := NewParam(splitParameter(lex.v))
			sig.Params = append(sig.Params, p)
		}
	}
	return
}

func splitParameter(s string) (name, t string, isPtr bool) {
	s = strings.TrimPrefix(s, "const ")
	splits := strings.Split(s, " ")
	name = splits[len(splits)-1]
	if renamed, ok := renames[name]; ok {
		name = renamed
	}
	if len(splits) > 2 {
		t = strings.Join(splits[:len(splits)-1], " ")
	} else {
		t = splits[0]
	}
	if isPtr = strings.HasSuffix(t, "*"); isPtr {
		t = t[:len(t)-1]
	}

	if fixed, ok := ctypesFix[t]; ok {
		t = fixed
	}
	return
}
