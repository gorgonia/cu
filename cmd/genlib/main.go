package main

import (
	"log"
	"os"
	"os/exec"
	"path"
	"strings"
)

var pkgloc string
var apiFile string
var ctxFile string

func init() {
	gopath := os.Getenv("GOPATH")
	pkgloc = path.Join(gopath, "src/github.com/chewxy/cu")
	apiFile = path.Join(pkgloc, "api.go")
	ctxFile = path.Join(pkgloc, "ctx_api.go")
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

func main() {
	input := strings.NewReader(src)
	sigs := Parse(input)

	var gss []*GoSignature
	sigs = filterCSigs(sigs)
	for _, sig := range sigs {
		gs := sig.GoSig()
		gss = append(gss, gs)
	}

	// generateAPIFile(gss)
	generateContextFile(gss)

	var err error
	filename := apiFile
	cmd := exec.Command("goimports", "-w", filename)
	if err = cmd.Run(); err != nil {
		log.Fatalf("Go imports failed with %v for %q", err, filename)
	}

	filename = ctxFile
	cmd = exec.Command("goimports", "-w", filename)
	if err = cmd.Run(); err != nil {
		log.Fatalf("Go imports failed with %v for %q", err, filename)
	}
}
