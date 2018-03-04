package main

import "text/template"

var alphaTemplateRaw = `var {{range .Params }}{{.}}C, {{end}} unsafe.Pointer
	if {{.Check}} == Float {
		var {{range .Params}} {{.}}F, {{end}} C.float
		{{range .Params -}} 
		{{.}}F = C.float(float32({{.}}))
		{{end -}}

		{{range .Params -}}
		{{.}}C = unsafe.Pointer(&{{.}}F)
		{{end -}}
	} else {
		var {{range .Params}} {{.}}F, {{end}} C.double
		{{range .Params -}} 
		{{.}}F = C.double({{.}})
		{{end -}}

		{{range .Params -}}
		{{.}}C = unsafe.Pointer(&{{.}}F)
		{{end -}}
	}
`

type Con struct {
	Ctype     string
	GoType    string
	Create    string
	Set       string
	Params    []string
	ParamType []string
}

var constructStrucRaw = `type {{.GoType}} struct {
	internal C.{{.Ctype}}

	{{$pt := .ParamType}}
	{{range $i, $v := .Params -}}
	{{unexport $v}} {{index $pt $i}}
	{{end -}}
}	
`

var constructionRaw = `var internal C.{{.Ctype}}
	if err := result(C.{{.Create}}(&internal)); err != nil {
		return nil, err
	}

	{{$pt := .ParamType}}
	if err := result(C.{{.Set}}(internal, {{range $i, $v := .Params -}}{{$t := index $pt $i -}}{{toC $v $t -}},{{end -}})); err != nil {
		return nil, err
	}

	return &{{.GoType}} {
		internal: internal,
		{{range .Params -}}
		{{.}}: {{.}},
		{{end -}}
	}, nil
`

var (
	alphaTemplate           *template.Template
	constructionTemplate    *template.Template
	constructStructTemplate *template.Template
)

var funcs = template.FuncMap{
	"isBuiltin": isBuiltin,
	"unexport":  unexport,
	"toC":       toC,
}

func init() {
	alphaTemplate = template.Must(template.New("alpha").Parse(alphaTemplateRaw))
	constructionTemplate = template.Must(template.New("cons").Funcs(funcs).Parse(constructionRaw))
	constructStructTemplate = template.Must(template.New("cons2").Funcs(funcs).Parse(constructStrucRaw))
}
