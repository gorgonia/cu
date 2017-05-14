package main

import "unicode"

type stateFn func(*Lexer) stateFn

func lexRetType(l *Lexer) (fn stateFn) {
	var next rune
	for next = l.next(); !unicode.IsSpace(next) && next != eof; next = l.next() {
		l.accept()
	}
	if next == eof {
		return nil
	}

	l.emit(RetType)
	l.ignore() // clear the space it read
	next = l.next()
	if next == eof {
		return nil
	}
	l.backup()
	return lexIdentifier
}

func lexIdentifier(l *Lexer) (fn stateFn) {
	for next := l.next(); next != '('; next = l.next() {
		l.accept()
	}
	l.backup()
	l.emit(FunctionName)

	// consume and ignore the brace
	l.next()
	l.ignore()
	return lexParam
}

func lexParam(l *Lexer) (fn stateFn) {
	for next := l.next(); next != ',' && next != ')'; next = l.next() {
		l.accept()
	}
	l.backup()
	l.emit(Parameter)

	// consume and ignore
	next := l.next()
	l.ignore()
	switch next {
	case ',':
		return lexParam
	case ')':
		l.next()
		return lexSemicolon
	}
	return lexWhitespace
}

func lexSemicolon(l *Lexer) (fn stateFn) {
	next := l.next()
	switch {
	case unicode.IsSpace(next):
		l.ignore()
		return lexRetType
	}
	return nil
}

func lexWhitespace(l *Lexer) (fn stateFn) {
	l.acceptRunFn(unicode.IsSpace)
	l.lineCount()
	l.ignore()

	return lexParam
}
