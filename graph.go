package cu

// #include <cuda.h>
import "C"

// Graph represents a CUDA graph
type Graph struct{}

func (g *Graph) c() {}

// Node represents a CUDA graph node
type Node struct{}

func (n *Node) c() {}

// ExecGraph represents a CUDA execution graph.
type ExecGraph struct{}

func (g *ExecGraph) c() {}
