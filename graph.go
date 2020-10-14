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
	var fromPtr, toPtr unsafe.Pointer
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
func (g Graph) AddHostNode(children []Node) (Node, error) {
	ptr, numDependencies := unsplatNodes(children)
	var retVal Node
	err := result(C.cuGraphAddHostNode(&retVal.n, g.c(), ptr, numDependencies, nil)) // TMP TODO wrt nil
	return retVal, err
}

// AddKernelNode creates a kernel execution node and adds it to the graph.
// When the graph is launched, the node will invoke the specified kernel function.
func (g Graph) AddKernelNode(children []Node, kernelParams []KernelParams) (Node, error) {
	ptr, numDependencies := unsplatNodes(children)
	var retVal Node
	err := result(C.cuGraphAddKernelNode(&retVal.n, g.c(), ptr, numDependencies, nil)) // TMP TODO
	return retVal, err
}

func (g Graph) Edges(from, to []Node) (edges []int, numEdges int, err error) {
	if len(from) != len(to) {
		return errors.Errorf("Expected from and to to have the same length. From is %d long. To is %d long", len(from), len(to))
	}
	if len(from) == 0 {
		return nil, 0, nil // TODO
	}

}

// Node represents a CUDA graph node
type Node struct{ n C.CUgraphNode }

func (n Node) c() C.CUgraphNode { return n.n }
func (n Node) String() string   { return fmt.Sprintf("Node_0x%x", uintptr(unsafe.Pointer(n.n))) }

// Destroy destroys the node.
func (n Node) Destroy() error { return result(C.cuGraphDestroyNode) }

// AddChild creates a child node, which executes an embedded graph and adds it to `in`.
// The result is a new node in the `in` graph, and a handle to that child node will be returned.
//
// The childGraph parameter is the graph to clone into the node.
func (n Node) AddChild(in Graph, children []Node, childGraph Graph) (Node, error) {
	var numDependencies C.size_t
	numDependencies = C.size_t(len(children))
	var retVal Node
	var ptr unsafe.Pointer
	if len(children) > 0 {
		ptr = unsafe.Pointer(&children[0])
	}
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
func unsplatNodes(a []Node) (ptr unsafe.Pointer, size C.size_t) {
	size = C.size_t(len(a))
	if len(a) > 0 {
		ptr = unsafe.Pointer(&a[0])
	}
	return ptr
}
