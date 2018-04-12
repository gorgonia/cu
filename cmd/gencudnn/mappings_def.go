package main

var fnNameMap map[string]string
var enumMappings map[string]string

// contextual is a list of functions that are contextual (i.e. they require the handle)
var contextual map[string]struct{}

// retVals is a list of functions that have return values.
var retVals map[string]map[int]string

// creations is a list of functions that creates shit. The key is the type
var creations map[string][]string

// setFns is a list of functions that sets types. The key is the type
var setFns map[string][]string

var destructions map[string][]string

// alphaBetas is a list of functions that have alphas, betas, in the parameters
var alphaBetas map[string]map[int]string

// memories is a list of functions that require memory in the parameters
var memories map[string]map[int]string

// methods enumerates the methods
var methods map[string][]string

// orphaned
var orphaned map[string]struct{}

var generated = map[string]struct{}{}
