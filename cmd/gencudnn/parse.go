package main

import (
	"fmt"
	"strings"

	"github.com/gorgonia/bindgen"
	"modernc.org/cc"
)

// Functions returns the C function declarations in the givel set of file paths.
func functions(decl *cc.Declarator) bool {
	if !strings.HasPrefix(bindgen.NameOf(decl), "cudnn") {
		return false
	}
	if decl.Type.Kind() == cc.Function {
		return true
	}
	return false

}

func enums(decl *cc.Declarator) bool {
	if !strings.HasPrefix(bindgen.NameOf(decl), "cudnn") {
		return false
	}
	if decl.Type.Kind() == cc.Enum {
		return true
	}
	return false
}

func otherTypes(decl *cc.Declarator) bool {
	if !strings.HasPrefix(bindgen.NameOf(decl), "cudnn") {
		return false
	}
	if decl.Type.Kind() == cc.Struct || decl.Type.Kind() == cc.Ptr {
		return true
	}
	return false

}

func isIgnored(a string) bool {
	if _, ok := ignoredEnums[a]; ok {
		return true
	}
	if _, ok := ignored[a]; ok {
		return true
	}
	if _, ok := ignoredTypes[a]; ok {
		return true
	}
	return false
}

func isContextual(a string) bool {
	if _, ok := contextual[a]; ok {
		return true
	}
	return false
}

func isBuiltin(a string) bool {
	if _, ok := builtins[a]; ok {
		return true
	}
	return false
}

func isPointerOfBuiltin(a string) (isPtr, isbuiltin bool) {
	if _, ok := builtins[a]; ok {
		return false, true
	}
	dp := depointerize(a)
	if _, ok := builtins[dp]; ok {
		return true, true
	}
	return dp == a, false
}

func processNameBasic(str string) string {
	return strings.TrimSuffix(strings.TrimPrefix(str, "cudnn"), "_t")
}

func nameOfType(a cc.Type) string {
	td := bindgen.TypeDefOf(a)
	if td != "" {
		return td
	}
	if bindgen.IsConstType(a) {
		return strings.TrimPrefix(a.String(), "const ")
	}
	return a.String()
}

func processEnumName(lcp, name string) string {
	// special cornercases names
	switch name {
	case "CUDNN_32BIT_INDICES":
		return "Indices32"
	case "CUDNN_64BIT_INDICES":
		return "Indices64"
	case "CUDNN_16BIT_INDICES":
		return "Indices16"
	case "CUDNN_8BIT_INDICES":
		return "Indices8"
	case "CUDNN_POOLING_MAX":
		return "MaxPooling"
	case "CUDNN_LRN_CROSS_CHANNEL_DIM1":
		return "CrossChannelDim1"
	case "CUDNN_DIVNORM_PRECOMPUTED_MEANS":
		return "PrecomputedMeans"
	case "CUDNN_SAMPLER_BILINEAR":
		return "Bilinear"
	case "CUDNN_PTR_16B_ALIGNED": // processing would yield `16B_Aligned`, which is not a valid Go name
		return "Ptr16"
	case "CUDNN_PTR_NULL":
		return "NullPtr"
	case "CUDNN_PTR_ELEM_ALIGNED":
		return "PtrElemAligned"
	case "CUDNN_BATCHNORM_OPS_BN":
		return "BatchNorm" //  name == lcp otherwise
	case "CUDNN_GENSTATS_SUM_SQSUM":
		return "SumSq" // it is the only enum in the list, so name == lcp
	case "CUDNN_NORM_OPS_NORM":
		return "Norm" // name == lcp otherwise
	}

	var trimmed string
	if len(lcp) < 9 && len(lcp) > 6 { // CUDNN_ or CUDNN_XXX
		lcp = "CUDNN_"
	}
	trimmed = strings.TrimPrefix(name, lcp)
	lowered := strings.ToLower(trimmed)

	switch lcp {
	case "CUDNN_RNN_CLIP_":
		lowered = "RNNClip" + strings.Title(lowered)
	case "CUDNN_RNN_":
		lowered = "RNN" + strings.Title(lowered)
	case "CUDNN_RNN_ALGO_":
		lowered = strings.Title(lowered) + "RNN"
	case "CUDNN_POINTWISE_":
		lowered = "Pointwise" + strings.Title(lowered)
	case "CUDNN_OP_TENSOR_":
		lowered = "Tensor" + strings.Title(lowered)
	case "CUDNN_NORM_OPS_NORM":
		lowered = "Norm" + strings.Title(lowered)
	case "CUDNN_NORM_PER_":
		lowered = "NormPer" + strings.Title(lowered)
	case "CUDNN_NORM_ALGO_":
		lowered = strings.Title(lowered) + "Norm"
	case "CUDNN_LOSS_NORMALIZATION_":
		lowered = "LossNorm" + strings.Title(lowered)
	case "CUDNN_BATCHNORM_OPS_BN":
		lowered = "BatchNorm" + strings.Title(lowered)
	case "CUDNN_LAYOUT_TYPE_":
		lowered = "BELayout" + strings.Title(lowered)
	case "CUDNN_BACKEND_":
		lowered = "BEDescriptor" + strings.Title(lowered)
	case "CUDNN_ATTR_":
		lowered = "BEAttrName" + strings.Title(lowered)
	case "CUDNN_TYPE_":
		lowered = "BEAttr" + strings.Title(lowered)
	case "CUDNN_TENSOR_N":
		// tensor description
		lowered = "n" + lowered
		upper := strings.ToUpper(lowered[:4])
		lowered = upper + lowered[4:]
	case "CUDNN_REDUCE_TENSOR_":
		// reduction op
		lowered = "Reduce_" + lowered
	case "CUDNN_CTC_LOSS_ALGO_":
		// CTC Loss Algorithms
		lowered = lowered + "CTCLoss"
	case "CUDNN_PTR_":
	default:
	}

	retVal := bindgen.Snake2Camel(lowered, true)

	// final cleanup
	switch retVal {
	case "Relu":
		return "ReLU"
	case "ClippedRelu":
		return "ClippedReLU"
	case "RnnRelu":
		return "RNNReLU"
	case "RnnTanh":
		return "RNNTanh"
	case "Lstm":
		return "LSTM"
	case "Gru":
		return "GRU"
	}
	return retVal
}

func searchByName(decls []bindgen.Declaration, name string) bindgen.Declaration {
	for _, d := range decls {
		if bindgen.NameOf(d.Decl()) == name {
			return d
		}
	}
	return nil
}

func unexport(a string) string {
	if a == "" {
		return ""
	}

	return strings.ToLower(string(a[0])) + a[1:]
}

func safeParamName(a string) string {
	if a == "C" {
		return "C_"
	}
	return a
}

func goNameOf(a cc.Type) string {
	n := nameOfType(a)
	return goNameOfStr(n)
}

// same as above, but given a c name type in string
func goNameOfStr(n string) (retVal string) {
	var ok bool
	defer func() {
		retVal = reqPtr(retVal)
	}()
	if retVal, ok = ctypes2GoTypes[n]; ok {
		return retVal
	}
	if retVal, ok = enumMappings[n]; ok {
		return retVal
	}
	if retVal, ok = builtins[n]; ok {
		return retVal
	}
	if retVal, ok = nonPrimitives[n]; ok {
		return retVal
	}

	return ""
}

func toC(name, typ string) string {
	if inList(strings.TrimRight(name, "C"), alphaBetaParams) {
		return name
	}

	for _, v := range enumMappings {
		if v == typ {
			return name + ".C()"
		}
	}

	for _, v := range ctypes2GoTypes {
		if v == typ || typ == "*"+v {
			return name + ".internal"
		}
	}

	if v, ok := go2cBuiltins[typ]; ok {
		return fmt.Sprintf("C.%v(%v)", v, name)

	}

	if typ == "Memory" {
		return fmt.Sprintf("unsafe.Pointer(%v.Uintptr())", name)
	}

	// log.Printf("name %q typ %q", name, typ)
	// panic("Unreachable")
	return name
}

func toCType(goType string) string {
	if v, ok := go2cBuiltins[goType]; ok {
		return v
	}
	if v, ok := go2cNonPrimitives[goType]; ok {
		return v
	}
	return "TODO"
}

func getRetVal(cs *bindgen.CSignature) map[int]string {

	name := cs.Name
	outputs := outputParams[name]
	ios := ioParams[name]
	if len(outputs)+len(ios) == 0 {
		return nil
	}

	retVal := make(map[int]string)
	for i, p := range cs.Parameters() {
		param := p.Name()
		if inList(param, outputs) || inList(param, ios) {
			retVal[i] = param
		}
	}
	return retVal
}

func getRetValOnly(cs *bindgen.CSignature) map[int]string {
	name := cs.Name
	outputs := outputParams[name]

	if len(outputs) == 0 {
		return nil
	}

	retVal := make(map[int]string)
	for i, p := range cs.Parameters() {
		param := p.Name()
		if inList(param, outputs) {
			retVal[i] = param
		}
	}
	return retVal
}

func depointerize(typeName string) string {
	return strings.TrimSuffix(strings.TrimPrefix(typeName, "*"), "*")
}

func reqPtr(gotyp string) string {
	for _, v := range ctypes2GoTypes {
		if v == gotyp {
			return "*" + gotyp
		}
	}
	return gotyp
}

func alreadyGenIn(name string, ins ...map[string][]string) bool {
	for _, in := range ins {
		for _, vs := range in {
			if inList(name, vs) {
				return true
			}
		}
	}
	return false
}

func alreadyDeclaredType(name string, ins ...map[string]string) bool {
	for _, in := range ins {
		if _, ok := in[name]; ok {
			return true
		}
	}
	return false
}

func inList(a string, list []string) bool {
	for _, v := range list {
		if a == v {
			return true
		}
	}
	return false
}
