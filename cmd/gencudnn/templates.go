package main

import "text/template"

var alphaTemplateRaw = `{{$lso := .LSO -}}
var {{range $i, $p := .Params }}{{$p}}C {{if lt $i $lso}},{{end}} {{end }} unsafe.Pointer
	switch {{.Check}}.dataType {
	case Float, Half:
		var {{range $i, $p :=  .Params }}{{$p}}F{{if lt $i $lso}},{{end }} {{end }} C.float 
		{{range .Params -}} 
		{{.}}F = C.float(float32({{.}}))
		{{end -}}

		{{range .Params -}}
		{{.}}C = unsafe.Pointer(&{{.}}F)
		{{end -}}
	case Double:
		var {{range $i, $p :=  .Params }}{{$p}}F{{if lt $i $lso}},{{end }} {{end }} C.double
		{{range .Params -}} 
		{{.}}F = C.double({{.}})
		{{end -}}

		{{range .Params -}}
		{{.}}C = unsafe.Pointer(&{{.}}F)
		{{end -}}
	default:
		{{if .MultiReturn -}}
		err = errors.Errorf("Unsupported data type: %v", {{.Check}}.dataType)
		return
		{{else -}}
		return errors.Errorf("Unsupported data type: %v", {{.Check}}.dataType) 
		{{end -}}
	}
`

type AlphaBeta struct {
	Params      []string // parameters that are alpha/beta
	Check       string   // what to check
	LSO         int      // length of params -1
	MultiReturn bool
}

var callTemplateRaw = `// call {{.CFuncName}}
{{if .MultiReturn -}} err = result(C.{{.CFuncName}}({{range $i, $v := .Params}}{{if $v.IsPtr}}&{{end}}{{toC $v.Name $v.Type -}}, {{end -}}))
{{else -}}return result(C.{{.CFuncName}}({{range $i, $v := .Params}}{{toC $v.Name $v.Type -}}, {{end -}})) {{end -}}
`

type Call struct {
	Params      []Param
	CFuncName   string
	MultiReturn bool
}

type Con struct {
	Ctype     string
	GoType    string
	Create    string
	Set       []string
	Destroy   string
	Params    []string
	ParamType []string
	TODO      string // TODO represents a part where human intervention is required
}

var constructStructRaw = `// {{.GoType}} is a representation of {{.Ctype}}. 
type {{.GoType}} struct {
	internal C.{{.Ctype}}

	{{$l := len .Set}}
	{{if gt $l 1}} //TODO 
	{{else -}}
	{{$pt := .ParamType}}
	{{range $i, $v := .Params -}}
	{{unexport $v}} {{index $pt $i}}
	{{end -}}
	{{end -}}
}	
`

var constructionRaw = `var internal C.{{.Ctype}}
	if err := result(C.{{.Create}}(&internal)); err != nil {
		return nil, err
	}

	{{if ne .TODO ""}}// TODO: {{.TODO}}{{end}}

	{{$pt := .ParamType}}
	if err := result(C.{{index .Set 0}}(internal, {{range $i, $v := .Params -}}{{$t := index $pt $i -}}{{toC $v $t -}},{{end -}})); err != nil {
		return nil, err
	}


	retVal =  &{{.GoType}} {
		internal: internal,
		{{range .Params -}}
		{{.}}: {{.}},
		{{end -}}
	}
	runtime.SetFinalizer(retVal, destroy{{.GoType}})
	return retVal, nil
`

var constructionTODORaw = `// available "Set" methods: 
{{range .Set -}}
//	{{.}}
{{end -}}
return nil, errors.Errorf("TODO: Manual Intervention required")
`

var destructRaw = `C.{{.Destroy}}(obj.internal)`

var (
	alphaTemplate            *template.Template
	callTemplate             *template.Template
	constructionTemplate     *template.Template
	constructionTODOTemplate *template.Template
	constructStructTemplate  *template.Template
	destructTemplate         *template.Template
)

var funcs = template.FuncMap{
	"isBuiltin": isBuiltin,
	"unexport":  unexport,
	"toC":       toC,
}

func init() {
	alphaTemplate = template.Must(template.New("alpha").Parse(alphaTemplateRaw))
	callTemplate = template.Must(template.New("call").Funcs(funcs).Parse(callTemplateRaw))
	constructionTemplate = template.Must(template.New("cons").Funcs(funcs).Parse(constructionRaw))
	constructionTODOTemplate = template.Must(template.New("cons").Funcs(funcs).Parse(constructionTODORaw))
	constructStructTemplate = template.Must(template.New("cons2").Funcs(funcs).Parse(constructStructRaw))
	destructTemplate = template.Must(template.New("Destroy").Funcs(funcs).Parse(destructRaw))
}
