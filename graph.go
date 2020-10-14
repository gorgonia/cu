package cu

// #include <cuda.h>
import "C"
import (
	"fmt"
	"unsafe"

	"github.com/pkg/errors"
)

// Graph represents a CUDA graph
type Graph struct{ g C.CUgraph }

func (g Graph) c() C.CUgraph   { return g.g }
func (g Graph) String() string { return fmt.Sprintf("Graph_0x%x", uintptr(unsafe.Pointer(g.g))) }

// MakeGraph makes a new graph.
func MakeGraph() (Graph, error) {
	var g Graph
	err := result(C.cuGraphCreate(&g.g, C.uint(0))) // flags must be 0
	return g, err
}

func (g Graph) Destroy() error { return result(C.cuGraphDestroy(g.g)) }

func (g Graph) Clone() (Graph, error) {
	var c Graph
	err := result(C.cuGraphClone(&c.g, g.g))
	return c, err
}

// AddDependencies adds edges to the graph. Both `from` and `to` must be the same length.
// An edge will be added from from[i] to to[i]
// If an edge already exists between the nodes, then an error will be returned
func (g Graph) AddDependencies(from, to []Node) error {
	if len(from) != len(to) {
		return errors.Errorf("Expected from and to to have the same length. From is %d long. To is %d long", len(from), len(to))
	}
	if len(from) == 0 {
		return nil
	}
	var numDependencies C.size_t
	var fromPtr, toPtr *C.CUgraphNode
	fromPtr, numDependencies = unsplatNodes(from)
	toPtr, _ = unsplatNodes(to)
	return result(C.cuGraphAddDependencies(g.c(), fromPtr, toPtr, numDependencies))
}

// AddEmptyNode creates an empty node and edds it to the graph. An empty node is a node that performs no operations during execution. It can be used for transitive ordering.
//  For example, a phased execution graph with 2 groups of n nodes with a barrier between them can be represented using an empty node and 2*n dependency edges, rather than no empty node and n^2 dependency edges.
func (g Graph) AddEmptyNode(children []Node) (Node, error) {
	ptr, numDependencies := unsplatNodes(children)
	var retVal Node
	err := result(C.cuGraphAddEmptyNode(&retVal.n, g.c(), ptr, numDependencies))
	return retVal, err
}

// AddHostNode creates a host execution node and adds it to the graph.
// When the graph is launched, the node will invoke the specified CPU function. Host nodes are not supported under MPS with pre-Volta GPUs.
func (g Graph) AddHostNode(children []Node, params *HostNodeParams) (Node, error) {
	ptr, numDependencies := unsplatNodes(children)
	var retVal Node
	err := result(C.cuGraphAddHostNode(&retVal.n, g.c(), ptr, numDependencies, params.c()))
	return retVal, err
}

// AddKernelNode creates a kernel execution node and adds it to the graph.
// When the graph is launched, the node will invoke the specified kernel function.
//
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g50d871e3bd06c1b835e52f2966ef366b
func (g Graph) AddKernelNode(children []Node, params *KernelNodeParams) (Node, error) {
	ptr, numDependencies := unsplatNodes(children)
	var retVal Node
	err := result(C.cuGraphAddKernelNode(&retVal.n, g.c(), ptr, numDependencies, params.c()))
	return retVal, err
}

// AddMemcpyNode creates a node which performs memcpy.
//
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g674da6ab54a677f13e0e0e8206ff5073
func (g Graph) AddMemcpyNode(children []Node, params *CopyParams, ctx Context) (Node, error) {
	ptr, numDependencies := unsplatNodes(children)
	var retVal Node
	err := result(C.cuGraphAddMemcpyNode(&retVal.n, g.c(), ptr, numDependencies, params.c(), ctx.CUDAContext().c()))
	return retVal, err
}

func (g Graph) AddMemsetNode(children []Node, params *MemsetParams, ctx Context) (Node, error) {
	ptr, numDependencies := unsplatNodes(children)
	var retVal Node
	err := result(C.cuGraphAddMemsetNode(&retVal.n, g.c(), ptr, numDependencies, params.c(), ctx.CUDAContext().c()))
	return retVal, err
}

// Edges returns the edges between nodes. CUDA's API is quite dodgy and unclear. It is reproduced below:
//
// Returns a list of hGraph's dependency edges. Edges are returned via corresponding indices in from and to; that is, the node in to[i] has a dependency on the node in from[i]. from and to may both be NULL, in which case this function only returns the number of edges in numEdges. Otherwise, numEdges entries will be filled in. If numEdges is higher than the actual number of edges, the remaining entries in from and to will be set to NULL, and the number of edges actually returned will be written to numEdges.
func (g Graph) Edges(from, to []Node) (edges []int, numEdges int, err error) {
	if len(from) != len(to) {
		return nil, -1, errors.Errorf("Expected from and to to have the same length. From is %d long. To is %d long", len(from), len(to))
	}
	if len(from) == 0 {
		return nil, 0, nil // TODO
	}
	retVal := make([]C.size_t, len(from))
	retVal[0] = C.size_t(len(from))

	fromPtr, _ := unsplatNodes(from)
	toPtr, _ := unsplatNodes(to)
	retPtr := (*C.size_t)(unsafe.Pointer(&retVal[0]))
	if err = result(C.cuGraphGetEdges(g.g, fromPtr, toPtr, retPtr)); err != nil {
		return nil, -1, err
	}
	numEdges = len(from)
	if len(from) == 0 {
		return
	}

	edges = make([]int, len(retVal))
	for i := range retVal {
		edges[i] = int(retVal[i])
	}
	return
}

// Node represents a CUDA graph node
type Node struct{ n C.CUgraphNode }

func (n Node) c() C.CUgraphNode { return n.n }
func (n Node) String() string   { return fmt.Sprintf("Node_0x%x", uintptr(unsafe.Pointer(n.n))) }

// Destroy destroys the node.
func (n Node) Destroy() error { return result(C.cuGraphDestroyNode(n.n)) }

// AddChild creates a child node, which executes an embedded graph and adds it to `in`.
// The result is a new node in the `in` graph, and a handle to that child node will be returned.
//
// The childGraph parameter is the graph to clone into the node.
func (n Node) AddChild(in Graph, children []Node, childGraph Graph) (Node, error) {
	var retVal Node
	ptr, numDependencies := unsplatNodes(children)
	err := result(C.cuGraphAddChildGraphNode(&retVal.n, in.c(), ptr, numDependencies, childGraph.c()))
	return retVal, err
}

// ExecGraph represents a CUDA execution graph.
type ExecGraph struct{ g C.CUgraphExec }

func (g ExecGraph) c() C.CUgraphExec { return g.g }
func (g ExecGraph) String() string {
	return fmt.Sprintf("ExecGraph_0x%x", uintptr(unsafe.Pointer(g.g)))
}

// Destroy destroys the execution graph.
func (g ExecGraph) Destroy() error { return result(C.cuGraphExecDestroy(g.g)) }

/* utility functions */

// unsplatNodes takes a Go slice and converts it to pointers and size so that it can be passed into C.
//
// This works because a Node is just an empty struct around a C.cuGraphNode (i.e. they are the same size). If Node's definition change, then we will have to allocate a new slice, copy the C.cuGraphNode into that slice, then unsplat.
func unsplatNodes(a []Node) (cunode *C.CUgraphNode, size C.size_t) {
	size = C.size_t(len(a))
	if len(a) > 0 {
		cunode = (*C.CUgraphNode)(unsafe.Pointer(&a[0]))
		return cunode, size
	}
	return nil, 0
}
