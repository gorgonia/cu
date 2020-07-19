package main

import (
	"log"
	"os"
	"os/exec"
	"path"
)

var pkgloc string
var apiFile string
var ctxFile string
var resultFile string

func init() {
	gopath := os.Getenv("GOPATH")
	pkgloc = path.Join(gopath, "src/gorgonia.org/cu")
	apiFile = path.Join(pkgloc, "api.go")
	ctxFile = path.Join(pkgloc, "ctx_api.go")
	resultFile = path.Join(pkgloc, "result.go")
}

func generateAPIFile(gss []*GoSignature) {
	f, err := os.Create(apiFile)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	f.WriteString(header)
	generateAPI(f, gss)
}

func generateContextFile(gss []*GoSignature) {
	g, err := os.Create(ctxFile)
	if err != nil {
		panic(err)
	}
	defer g.Close()
	g.WriteString(header)
	generateContextAPI(g, gss)
}

func generateResultFile() {
	g, err := os.Create(resultFile)
	if err != nil {
		panic(err)
	}
	defer g.Close()

	g.WriteString(resultHeader)
	generateResultEnums(g)
}

func main() {
	// input := strings.NewReader(src)
	// sigs := Parse(input)
	//sigs := Parse()

	//var gss []*GoSignature
	//sigs = filterCSigs(sigs)
	//for _, sig := range sigs {
	//		gs := sig.GoSig()
	//	gss = append(gss, gs)
	//}

	generateResultFile()
	//generateAPIFile(gss)
	//generateContextFile(gss)

	var err error
	files := []string{
		apiFile,
		ctxFile,
		resultFile,
	}

	for _, filename := range files {
		cmd := exec.Command("goimports", "-w", filename)
		if err = cmd.Run(); err != nil {
			log.Printf("Go imports failed with %v for %q", err, filename)
		}
	}

}
