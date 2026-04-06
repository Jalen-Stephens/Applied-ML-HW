# Homework 3 Part A: Learning on Graphs by Hand

**References:**
- Homework 3 Part A handout in file `HW3/Homework_3_Part_A_GNN.pdf`
- Lecture 15, Machine Learning on Graphs Part 3, on aggregation and update operators and node, edge, and graph-level tasks. File `Lecture_15_Graph_Neural_Networks_3.pdf`
- Lecture 16, GNNs for Scientific Simulation, on the encoder-processor-decoder pipeline and simulation as learning on graphs. File `Lecture_16_GNNs_for_Scientific_Simulation.pdf`
- Survey: *Theory of Graph Neural Networks: Representation and Learning* by Stefanie Jegelka

---

## Part I. Graphs, Matrices, and Neighborhood Aggregation

---

### Problem 1. Reading a graph as a matrix (10 pts)

We have the path graph on four nodes: \(1 - 2 - 3 - 4\).

#### (a) Adjacency matrix

Two nodes share an edge if and only if they are adjacent on the path. That gives us:

\[
A = \begin{bmatrix}
0 & 1 & 0 & 0 \\
1 & 0 & 1 & 0 \\
0 & 1 & 0 & 1 \\
0 & 0 & 1 & 0
\end{bmatrix}.
\]

#### (b) Compute \(Ax\)

The initial feature vector is \(x = (1, 0, 2, 1)^\top\). We multiply row by row:

\[
(Ax)_1 = 0(1) + 1(0) + 0(2) + 0(1) = 0,
\]
\[
(Ax)_2 = 1(1) + 0(0) + 1(2) + 0(1) = 1 + 2 = 3,
\]
\[
(Ax)_3 = 0(1) + 1(0) + 0(2) + 1(1) = 0 + 1 = 1,
\]
\[
(Ax)_4 = 0(1) + 0(0) + 1(2) + 0(1) = 2.
\]

So:

\[
Ax = \begin{bmatrix} 0 \\ 3 \\ 1 \\ 2 \end{bmatrix}.
\]

#### (c) Interpretation of \((Ax)_i\)

The \(i\)-th entry of \(Ax\) equals the sum of the feature values of node \(i\)'s immediate neighbors. For instance, \((Ax)_2 = x_1 + x_3 = 1 + 2 = 3\), which collects the features of nodes 1 and 3, the two neighbors of node 2. This matches the one-hop neighborhood aggregation that Lecture 15 describes as the core operation inside a GNN layer.

#### (d) Compute \(A^2 x\) and interpret

Rather than squaring \(A\) first, it is easier to compute \(A(Ax)\). We already know \(Ax = (0, 3, 1, 2)^\top\), so we just multiply by \(A\) again:

\[
(A^2 x)_1 = 0(0) + 1(3) + 0(1) + 0(2) = 3,
\]
\[
(A^2 x)_2 = 1(0) + 0(3) + 1(1) + 0(2) = 0 + 1 = 1,
\]
\[
(A^2 x)_3 = 0(0) + 1(3) + 0(1) + 1(2) = 3 + 2 = 5,
\]
\[
(A^2 x)_4 = 0(0) + 0(3) + 1(1) + 0(2) = 1.
\]

\[
A^2 x = \begin{bmatrix} 3 \\ 1 \\ 5 \\ 1 \end{bmatrix}.
\]

**Interpretation:** \(A^2\) counts length-2 walks. So \((A^2 x)_i\) aggregates features from nodes reachable in exactly two steps. For example, node 1 can reach node 3 via \(1 \to 2 \to 3\), so node 3's feature contributes to \((A^2 x)_1\). Node 1 can also reach itself via \(1 \to 2 \to 1\), so its own feature shows up too, with weight reflecting how many such walks exist. This is why stacking GNN layers deepens the receptive field: each additional multiplication by \(A\) extends the information horizon by one hop.

---

### Problem 2. One round of message passing by hand (10 pts)

Same graph \(1 - 2 - 3 - 4\), initial features \(h^{(0)} = (1, 0, 2, 1)^\top\).

The toy GNN update is:

\[
m_v^{(1)} = \sum_{u \in \mathcal{N}(v)} h_u^{(0)}, \qquad h_v^{(1)} = h_v^{(0)} + m_v^{(1)}.
\]

#### (a) Messages \(m_v^{(1)}\)

We look up each node's neighbors and sum their features:

- **Node 1:** \(\mathcal{N}(1) = \{2\}\). \(m_1^{(1)} = h_2^{(0)} = 0\).
- **Node 2:** \(\mathcal{N}(2) = \{1, 3\}\). \(m_2^{(1)} = h_1^{(0)} + h_3^{(0)} = 1 + 2 = 3\).
- **Node 3:** \(\mathcal{N}(3) = \{2, 4\}\). \(m_3^{(1)} = h_2^{(0)} + h_4^{(0)} = 0 + 1 = 1\).
- **Node 4:** \(\mathcal{N}(4) = \{3\}\). \(m_4^{(1)} = h_3^{(0)} = 2\).

#### (b) Updated representations \(h_v^{(1)}\)

Add each node's own feature to its message:

\[
h_1^{(1)} = 1 + 0 = 1, \quad
h_2^{(1)} = 0 + 3 = 3, \quad
h_3^{(1)} = 2 + 1 = 3, \quad
h_4^{(1)} = 1 + 2 = 3.
\]

#### (c) Result vector

\[
h^{(1)} = (1, 3, 3, 3)^\top.
\]

#### (d) Why this update only uses 1-hop information

The message \(m_v^{(1)}\) sums features of nodes in \(\mathcal{N}(v)\), which by definition are only the direct neighbors, meaning nodes at graph distance exactly 1. No neighbor-of-neighbor information enters the sum. So after one round, each node knows about itself and its immediate neighborhood, nothing further. This is exactly the "aggregate neighbors" step shown in Lecture 15's GNN pipeline diagram.

---

### Problem 3. Two rounds of message passing by hand (10 pts)

We continue from Problem 2 using \(h^{(1)} = (1, 3, 3, 3)^\top\):

\[
m_v^{(2)} = \sum_{u \in \mathcal{N}(v)} h_u^{(1)}, \qquad h_v^{(2)} = h_v^{(1)} + m_v^{(2)}.
\]

#### (a) Compute \(m_v^{(2)}\) and \(h_v^{(2)}\) for all nodes

Messages:

- **Node 1:** \(m_1^{(2)} = h_2^{(1)} = 3\).
- **Node 2:** \(m_2^{(2)} = h_1^{(1)} + h_3^{(1)} = 1 + 3 = 4\).
- **Node 3:** \(m_3^{(2)} = h_2^{(1)} + h_4^{(1)} = 3 + 3 = 6\).
- **Node 4:** \(m_4^{(2)} = h_3^{(1)} = 3\).

Updated representations:

\[
h_1^{(2)} = 1 + 3 = 4, \quad
h_2^{(2)} = 3 + 4 = 7, \quad
h_3^{(2)} = 3 + 6 = 9, \quad
h_4^{(2)} = 3 + 3 = 6.
\]

\[
h^{(2)} = (4, 7, 9, 6)^\top.
\]

#### (b) Which nodes can influence node 2 after two rounds?

After round 1, node 2 already incorporated information from its direct neighbors \(\{1, 3\}\). In round 2, it aggregates \(h_1^{(1)}\) and \(h_3^{(1)}\), which themselves already contain information from \(\{2\}\) and \(\{2, 4\}\) respectively. So node 4's original feature has now reached node 2 via the path \(4 \to 3 \to 2\). That means **all four nodes** \(\{1, 2, 3, 4\}\) can influence node 2 after two rounds.

#### (c) Why \(t\) rounds reach only the \(t\)-hop neighborhood

Think of it inductively. After round 1, each node's representation contains information from nodes at distance \(\leq 1\). In round 2, each node aggregates its neighbors' round-1 representations, which already encode distance-\(\leq 1\) information from those neighbors. So now each node's representation encodes information from distance \(\leq 2\). In general, each additional round extends the information horizon by exactly one hop, so after \(t\) rounds only nodes within graph distance \(t\) can have contributed. This is why the number of message-passing layers controls the receptive field.

---

### Problem 4. ReLU version of a manual GNN (10 pts)

Graph: \(1 - 2 - 3\), with \(h^{(0)} = (-1, 2, -2)^\top\).

Update rule:

\[
m_v^{(1)} = \sum_{u \in \mathcal{N}(v)} h_u^{(0)}, \qquad h_v^{(1)} = \operatorname{ReLU}\!\bigl(h_v^{(0)} + m_v^{(1)}\bigr),
\]

where \(\operatorname{ReLU}(z) = \max\{z, 0\}\).

#### (a) Messages

- **Node 1:** \(\mathcal{N}(1) = \{2\}\). \(m_1^{(1)} = h_2^{(0)} = 2\).
- **Node 2:** \(\mathcal{N}(2) = \{1, 3\}\). \(m_2^{(1)} = h_1^{(0)} + h_3^{(0)} = (-1) + (-2) = -3\).
- **Node 3:** \(\mathcal{N}(3) = \{2\}\). \(m_3^{(1)} = h_2^{(0)} = 2\).

#### (b) Updated representations with ReLU

First compute the pre-activation values:

\[
h_1^{(0)} + m_1^{(1)} = -1 + 2 = 1, \quad
h_2^{(0)} + m_2^{(1)} = 2 + (-3) = -1, \quad
h_3^{(0)} + m_3^{(1)} = -2 + 2 = 0.
\]

Now apply ReLU:

\[
h_1^{(1)} = \operatorname{ReLU}(1) = 1, \quad
h_2^{(1)} = \operatorname{ReLU}(-1) = 0, \quad
h_3^{(1)} = \operatorname{ReLU}(0) = 0.
\]

\[
h^{(1)} = (1, 0, 0)^\top.
\]

#### (c) How the nonlinearity changes things

If we use the same linear update as in Problems 2 and 3 without ReLU, every aggregated value, positive or negative, passes through unchanged. With ReLU, negative pre-activation values get clipped to zero. First, that introduces nonlinearity, so the overall multi-layer GNN can represent more complex functions than a single linear map. Second, it creates sparsity. Nodes whose local neighborhood effectively cancels out can end up with a zero representation, which can help or hurt depending on the task. As Lecture 15 explains, a nonlinear activation such as ReLU is what makes stacking multiple GNN layers different from simply multiplying by \(A\) many times.

---

## Part II. Weisfeiler-Leman Color Refinement by Hand

---

### Problem 5. Run 1-WL on a path graph (10 pts)

Path graph: \(1 - 2 - 3 - 4\). All nodes initially get color \(a\).

#### (a) Colors at \(t = 0\)

\[
c_1^{(0)} = a, \quad c_2^{(0)} = a, \quad c_3^{(0)} = a, \quad c_4^{(0)} = a.
\]

#### (b) Round 1

For each node, we form a pair consisting of its own color together with the multiset of its neighbors' colors, and then we assign a fresh name to each distinct pair:

| Node | Own color | Neighbor multiset | Signature | New color |
|------|-----------|-------------------|-----------|-----------|
| 1 | \(a\) | \(\{a\}\) | \((a, \{a\})\) | \(b\) |
| 2 | \(a\) | \(\{a, a\}\) | \((a, \{a, a\})\) | \(c\) |
| 3 | \(a\) | \(\{a, a\}\) | \((a, \{a, a\})\) | \(c\) |
| 4 | \(a\) | \(\{a\}\) | \((a, \{a\})\) | \(b\) |

After round 1: colors are \((b, c, c, b)\).

#### (c) Round 2

Now we use the round-1 colors:

| Node | Own color | Neighbor multiset | Signature | New color |
|------|-----------|-------------------|-----------|-----------|
| 1 | \(b\) | \(\{c\}\) | \((b, \{c\})\) | \(d\) |
| 2 | \(c\) | \(\{b, c\}\) | \((c, \{b, c\})\) | \(e\) |
| 3 | \(c\) | \(\{c, b\}\) | \((c, \{b, c\})\) | \(e\) |
| 4 | \(b\) | \(\{c\}\) | \((b, \{c\})\) | \(d\) |

After round 2: colors are \((d, e, e, d)\).

#### (d) Has the coloring stabilized?

Yes. Round 1 produced two color classes, \(\{1, 4\}\) and \(\{2, 3\}\). Round 2 still splits the vertices into the same two groups, even though the labels for those groups are new. Since the partition of nodes into equivalence classes did not change, 1-WL has stabilized.

#### (e) Which nodes share a color and why?

Nodes 1 and 4 share a color, and nodes 2 and 3 share a different color. On the path \(1 - 2 - 3 - 4\), nodes 1 and 4 are both endpoints with degree 1, while nodes 2 and 3 are interior vertices with degree 2. The path has a reflection symmetry that swaps \(1 \leftrightarrow 4\) and \(2 \leftrightarrow 3\), and 1-WL captures exactly this automorphism.

---

### Problem 6. Run 1-WL on a cycle graph (10 pts)

Cycle graph \(C_4\): \(1 - 2 - 3 - 4 - 1\). All nodes start with color \(a\).

#### (a) Two rounds

**Round 1:** Every node has degree 2, so every node sees the same signature:

\[
(a, \{a, a\}) \quad \text{for all nodes}.
\]

Assign one new color, say \(b\). All four nodes get color \(b\).

**Round 2:** Now every node has color \(b\) and both of its neighbors also have color \(b\):

\[
(b, \{b, b\}) \quad \text{for all nodes}.
\]

Assign one new color, say \(c\). All four nodes get color \(c\).

#### (b) What happens?

The colors receive new names each round as labels move from \(a\) to \(b\) to \(c\), but the partition stays the same, and all nodes remain in one single color class. The algorithm has already stabilized after round 1.

#### (c) Why 1-WL cannot distinguish nodes in this graph

Every node in \(C_4\) has the exact same local structure: degree 2, neighbors also degree 2, neighbors' neighbors also degree 2, and so on. Since 1-WL only looks at the current color together with the multiset of neighbor colors, that combined descriptor is identical for every node at every step, so no refinement ever happens. The cycle is vertex-transitive, so there is an automorphism mapping any node to any other, and 1-WL reflects that symmetry.

---

### Problem 7. Compare two graphs using 1-WL (10 pts)

- \(G_1\): path graph \(P_4\) on nodes \(\{1, 2, 3, 4\}\).
- \(G_2\): cycle graph \(C_4\) on nodes \(\{1, 2, 3, 4\}\).

All nodes start with color \(a\).

#### (a) Run two rounds on both

We already did this in Problems 5 and 6.

**\(G_1 = P_4\):**
At round 0 every node has color \(a\), so the color tuple is \((a, a, a, a)\). After round 1 the path splits into endpoints versus interior nodes, giving \((b, c, c, b)\). After round 2 we have \((d, e, e, d)\), which is the same partition of vertices as before, only with fresh labels.

**\(G_2 = C_4\):**
At round 0 the tuple is again \((a, a, a, a)\). After round 1 every node shares one color, \((b, b, b, b)\), and after round 2 they still all agree, \((c, c, c, c)\).

#### (b) Multiset of node colors at each round

| Round | \(G_1 = P_4\) | \(G_2 = C_4\) |
|-------|----------------|----------------|
| 0 | \(\{a, a, a, a\}\) | \(\{a, a, a, a\}\) |
| 1 | \(\{b, b, c, c\}\) | \(\{b', b', b', b'\}\) |
| 2 | \(\{d, d, e, e\}\) | \(\{c', c', c', c'\}\) |

I write colors for \(G_2\) with primes so it is clear they are fresh labels that need not match the symbols used for \(G_1\).

#### (c) When can 1-WL tell them apart?

The graphs separate already at **round 1**. Graph \(G_1\) uses two distinct colors, \(b\) and \(c\), while \(G_2\) uses only one, \(b'\). The multisets \(\{b, b, c, c\}\) and \(\{b', b', b', b'\}\) differ because one records two color classes and the other records a single class, so 1-WL declares the graphs non-isomorphic after just one refinement step.

#### (d) Why comparing multisets is natural for graph isomorphism

If two graphs are truly isomorphic, then there is a bijection between their nodes that preserves adjacency. Under such a bijection, every node would get the same WL color as its image. So the multiset of colors must be identical for isomorphic graphs. Conversely, if the color multisets ever differ, the graphs cannot be isomorphic. Comparing multisets in this way gives a sound test for non-isomorphism, although it does not always succeed on every pair of non-isomorphic graphs.

---

### Problem 8. A graph pair that 1-WL struggles with (10 pts)

- \(G_3\) is the cycle \(C_6\), a single 6-cycle.
- \(G_4\) is the disjoint union of two triangles, written \(K_3 \sqcup K_3\).

#### (a) Degree of every node

In \(G_3 = C_6\): every node has degree 2.

In \(G_4 = K_3 \sqcup K_3\): every node has degree 2 because each vertex in a triangle is adjacent to the other two vertices in that triangle.

So the degree sequence is \((2, 2, 2, 2, 2, 2)\) for both graphs.

#### (b) Two rounds of 1-WL

All nodes start with color \(a\).

**Round 1:** In both graphs, every node has degree 2 and both neighbors have color \(a\). The signature for every node is:

\[
(a, \{a, a\}).
\]

All 6 nodes in both graphs get the same new color \(b\).

**Round 2:** Now every node has color \(b\) and both neighbors have color \(b\). The signature is:

\[
(b, \{b, b\}).
\]

All nodes get the same new color \(c\).

#### (c) Color pattern in each graph

Both \(G_3\) and \(G_4\) end up with a single color class of size 6: \(\{c, c, c, c, c, c\}\).

#### (d) Can 1-WL distinguish \(G_3\) and \(G_4\)?

**No.** The color multisets are identical at every round. 1-WL cannot tell a hexagon apart from two disjoint triangles.

#### (e) Why this is a warning for message passing GNNs

A key result in the Jegelka survey, which our handout cites, states that standard message passing GNNs are at most as powerful as 1-WL. If 1-WL fails to distinguish two graphs, then no injective sum-based or mean-based MPNN can distinguish them either. Graph \(C_6\) is connected and contains a single 6-cycle, while \(K_3 \sqcup K_3\) is disconnected and consists of two triangles, so the two graphs are very different globally even though 1-WL sees them as the same. Any task that requires telling these graphs apart is beyond the reach of standard MPNNs, which motivates higher-order ideas such as \(k\)-WL or subgraph GNNs.

---

### Problem 9. Manual GNN versus 1-WL (10 pts)

Return to \(G_3 = C_6\) and \(G_4 = K_3 \sqcup K_3\). All nodes get initial feature \(h_v^{(0)} = 1\). The update rule is:

\[
h_v^{(t)} = h_v^{(t-1)} + \sum_{u \in \mathcal{N}(v)} h_u^{(t-1)}.
\]

#### (a) Compute \(h_v^{(1)}\) for every node in both graphs

Every node has degree 2, and all initial features are 1. So for any node \(v\) in either graph:

\[
h_v^{(1)} = h_v^{(0)} + \sum_{u \in \mathcal{N}(v)} h_u^{(0)} = 1 + 1 + 1 = 3.
\]

In \(G_3\): \(h^{(1)} = (3, 3, 3, 3, 3, 3)^\top\).

In \(G_4\): \(h^{(1)} = (3, 3, 3, 3, 3, 3)^\top\).

The two vectors are identical.

#### (b) Compute \(h_v^{(2)}\)

Now every node has feature 3 and degree 2:

\[
h_v^{(2)} = h_v^{(1)} + \sum_{u \in \mathcal{N}(v)} h_u^{(1)} = 3 + 3 + 3 = 9.
\]

In \(G_3\): \(h^{(2)} = (9, 9, 9, 9, 9, 9)^\top\).

In \(G_4\): \(h^{(2)} = (9, 9, 9, 9, 9, 9)^\top\).

Again the two graphs agree at every coordinate.

#### (c) Observation

The toy GNN produces exactly the same node representations for both graphs after every round. It cannot distinguish \(G_3\) from \(G_4\). This mirrors the 1-WL result from Problem 8.

#### (d) Intuitive connection between MPNNs and 1-WL

Both procedures work by iteratively updating each node based on a summary of its neighbors. The 1-WL algorithm hashes the current color together with the multiset of neighbor colors into a new color. The toy GNN adds the current feature to the sum of neighbor features. In both cases, the update depends on the local neighborhood in a way that does not depend on the order in which neighbors are listed. When two nodes have identical neighborhoods at round \(t-1\), they receive identical updates at round \(t\). So if 1-WL cannot distinguish two nodes or two graphs, then a sum-aggregation GNN without injective transforms cannot distinguish them either. This is the core of the classical expressiveness bound that says such GNNs are no stronger than 1-WL.

---

## Part III. Theory: What GNNs Can and Cannot Represent

---

### Problem 10. Permutation invariance (10 pts)

#### (a) What a node relabeling does

Relabeling the nodes means assigning new indices to the same set of vertices. Which pairs of vertices are connected does not change, only the numbering changes. If we apply a permutation \(\pi\) to the node indices, the adjacency matrix changes from \(A\) to \(PAP^\top\) for the corresponding permutation matrix \(P\), yet the underlying graph is still the same abstract object.

#### (b) Why a graph classifier should be invariant

The class of a graph, such as toxic versus non-toxic in a molecular property task, depends on its structure rather than on how we happened to number the atoms. If renaming nodes changed the output, the model would give inconsistent predictions for the same structure under different indexings. So a graph classification function should satisfy \(f(G) = f(\pi(G))\) for every permutation \(\pi\), which is what we mean by permutation invariance.

#### (c) A permutation-invariant readout

\[
\text{readout}(G) = \sum_{v \in V} h_v^{(T)}.
\]

Summing over all nodes is invariant because addition is commutative, so the order of the summands does not matter. Mean pooling \(\frac{1}{|V|}\sum_v h_v\) and max pooling \(\max_v h_v\) are also invariant.

#### (d) A readout that is NOT permutation invariant

\[
\text{readout}(G) = h_1^{(T)},
\]

In other words, we output only the representation of node 1. That depends on which vertex happens to be labeled 1, so relabeling would change the output. Concatenating node features in index order also fails, because permuting vertices reorders the concatenated vector.

---

### Problem 11. Why sum aggregation matters (optional challenge)

#### (a) What is a multiset?

A multiset is like a set except elements can appear more than once. For example, \(\{a, a, b\}\) is a multiset with \(a\) appearing twice. In an ordinary set, \(\{a, a, b\} = \{a, b\}\), but in a multiset the multiplicities matter.

#### (b) Why neighborhoods are multisets

When we aggregate neighbor features, multiple neighbors can have the same feature value. For instance, if nodes 3 and 5 both have feature vector \(h = [1, 0]\), the neighborhood of their shared neighbor is the multiset \(\{[1, 0], [1, 0], \ldots\}\), not a set. The aggregation must account for the fact that the same feature appears twice.

#### (c) Why sum is more natural than concatenation

Concatenation produces an output whose size depends on the degree of the node and it forces us to pick an order of neighbors. Summation is order-invariant, returns a fixed-size vector no matter how many neighbors there are, and naturally accumulates contributions from all neighbors. Those properties match what we want from permutation-invariant aggregation in a GNN.

#### (d) Why averaging can lose information that summing keeps

Consider two neighborhoods:

- Neighborhood A: features \(\{1, 1, 1\}\). Sum = 3, mean = 1.
- Neighborhood B: features \(\{1\}\). Sum = 1, mean = 1.

The means are identical, so a mean-aggregation GNN cannot tell these apart. But the sums are different (3 vs 1), so a sum-aggregation GNN can. The sum implicitly encodes the degree, meaning how many neighbors were summed, while the mean normalizes that count away. That distinction is one reason the Graph Isomorphism Network paper argues for sum aggregation when you want maximum expressive power within this framework.

---

### Problem 12. Computation trees and locality (optional challenge)

#### (a) Depth-2 computation tree rooted at node 2

For the path graph \(1 - 2 - 3 - 4\):

```
 [2] depth 0
 / \
 [1] [3] depth 1
 | / \
 [2] [2] [4] depth 2
```

At depth 0: node 2.
At depth 1: neighbors of 2, which are nodes 1 and 3.
At depth 2 we attach the neighbors of node 1, so another copy of node 2 appears, and we attach the neighbors of node 3, namely nodes 2 and 4 again.

#### (b) Which original nodes appear?

Every original vertex \(1, 2, 3,\) and \(4\) shows up somewhere in the tree. Node 2 appears at depth 0 and again twice at depth 2, but those appearances are separate copies in the unfolded tree rather than a single vertex being drawn twice in the original graph.

#### (c) Same local unfolding implies hard to distinguish

If two nodes have isomorphic computation trees with matching structure and leaf features, then after \(t\) rounds of message passing they will have identical representations. The GNN computes the same function on both trees. Even if those nodes sit in different global positions in the original graph, the GNN cannot tell them apart. This matches the limitation that the WL test formalizes.

#### (d) Why global properties are difficult

Global properties such as connectivity, diameter, or the number of cycles depend on structure that may lie far from any one node. A computation tree of depth \(t\) only captures the \(t\)-hop neighborhood around the root. If the relevant global structure lies outside every node's local tree, no finite depth of message passing can detect it by itself. For example, two nodes in different connected components have disjoint computation trees, yet deciding that the graph is disconnected really calls for a global readout rather than a purely local rule.

---

### Problem 13. Limits of message passing (optional bonus, 5 pts)

#### (a) Why local aggregation struggles with graph diameter

Graph diameter is the longest shortest path between any two nodes, which is a fundamentally global quantity. A node's \(t\)-hop neighborhood only sees a local ball of radius \(t\). To compute the diameter, you would need information about the most distant pair of nodes, which may be far from any single node's receptive field. Unless \(t\) is at least as large as the diameter itself, the message-passing representation at any single node cannot encode the full picture.

#### (b) Why counting certain cycles is hard

Counting six-cycles is subtle because any one vertex only sees a small piece of the cycle. In addition, 1-WL and the usual MPNNs tied to it cannot separate all regular graphs that share the same degree pattern, yet those graphs can still have different cycle structures, as we saw when comparing \(C_6\) to \(K_3 \sqcup K_3\). Purely local aggregation does not get a bird's-eye view of how a long cycle closes.

#### (c) A local property and a global property

Node degree is a good example of a local property, since you can read it off from a single neighborhood, and one round of message passing already captures it. Graph connectivity is a global property, because knowing whether the graph is connected or splits into several components requires reasoning about the whole graph, not one patch alone.

#### (d) Shallow GNN performance

A shallow GNN with only a few layers should do well when the answer depends mainly on a node's immediate surroundings, for example predicting a role from its local pattern of degrees and neighbors. I would expect it to struggle on tasks that hinge on connectivity or diameter, because those quantities really draw on the whole graph and cannot be inferred from a very small number of hops alone.

---

### Problem 14. Higher-order ideas (optional challenge)

#### (a) Why node pairs capture more structure

If we only keep one embedding per node, a GNN can only see what sits inside that node's local tree. Some relational facts, such as whether two specified nodes are joined by a short path or share a common neighbor, are really statements about pairs of vertices. By maintaining representations for unordered pairs or more generally for \(k\)-tuples, the model can encode that relational structure directly instead of forcing a single-node vector to imply it. Higher-order WL tests such as \(k\)-WL for \(k \geq 2\) are strictly more expressive than 1-WL, which illustrates the same idea.

#### (b) Main computational drawback

For \(k\)-tuples, the number of states scales as \(O(n^k)\) where \(n\) is the number of nodes. Even \(k = 2\) gives \(O(n^2)\) states, and each state must be updated at every round. For large graphs, this quickly becomes impractical in both memory and compute.

#### (c) The tradeoff in my own words

There is a direct tension between expressiveness and scalability. Single-node message passing is efficient (\(O(n)\) states) but has provable blind spots. Moving to pairs or triples dramatically increases what the model can distinguish, but at a steep computational cost. In practice, researchers look for middle-ground approaches like subgraph GNNs or local higher-order methods that gain expressiveness without the full \(O(n^k)\) blowup.

---

## Part IV. Spectral Graph Theory for Beginners

---

### Problem 15. Build the Laplacian (10 pts)

Path graph: \(1 - 2 - 3 - 4\).

#### (a) Adjacency matrix \(A\)

\[
A = \begin{bmatrix}
0 & 1 & 0 & 0 \\
1 & 0 & 1 & 0 \\
0 & 1 & 0 & 1 \\
0 & 0 & 1 & 0
\end{bmatrix}.
\]

#### (b) Degree matrix \(D\)

\(D\) is diagonal with \(D_{ii} = \deg(i)\):

- Node 1: degree 1.
- Node 2: degree 2.
- Node 3: degree 2.
- Node 4: degree 1.

\[
D = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 2 & 0 & 0 \\
0 & 0 & 2 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}.
\]

#### (c) Laplacian \(L = D - A\)

\[
L = D - A = \begin{bmatrix}
1 & -1 & 0 & 0 \\
-1 & 2 & -1 & 0 \\
0 & -1 & 2 & -1 \\
0 & 0 & -1 & 1
\end{bmatrix}.
\]

#### (d) Verify \(L \mathbf{1} = 0\)

Let \(\mathbf{1} = (1, 1, 1, 1)^\top\):

\[
L\mathbf{1} = \begin{bmatrix}
1(1) + (-1)(1) + 0(1) + 0(1) \\
(-1)(1) + 2(1) + (-1)(1) + 0(1) \\
0(1) + (-1)(1) + 2(1) + (-1)(1) \\
0(1) + 0(1) + (-1)(1) + 1(1)
\end{bmatrix}
= \begin{bmatrix}
1 - 1 \\ -1 + 2 - 1 \\ -1 + 2 - 1 \\ -1 + 1
\end{bmatrix}
= \begin{bmatrix} 0 \\ 0 \\ 0 \\ 0 \end{bmatrix}.
\]

This confirms that \(\mathbf{1}\) is an eigenvector of \(L\) with eigenvalue 0. Intuitively, if every node has the same value, there is no "disagreement" across any edge, so the Laplacian (which measures local differences) gives zero.

---

### Problem 16. Laplacian as smoothing difference operator (10 pts)

Let \(f = (1, 3, 3, 1)^\top\) on the path graph \(1 - 2 - 3 - 4\).

#### (a) Compute \(Lf\)

\[
(Lf)_1 = 1(1) + (-1)(3) + 0(3) + 0(1) = 1 - 3 = -2,
\]
\[
(Lf)_2 = (-1)(1) + 2(3) + (-1)(3) + 0(1) = -1 + 6 - 3 = 2,
\]
\[
(Lf)_3 = 0(1) + (-1)(3) + 2(3) + (-1)(1) = -3 + 6 - 1 = 2,
\]
\[
(Lf)_4 = 0(1) + 0(3) + (-1)(3) + 1(1) = -3 + 1 = -2.
\]

\[
Lf = \begin{bmatrix} -2 \\ 2 \\ 2 \\ -2 \end{bmatrix}.
\]

#### (b) Interpretation of each coordinate

Each entry \((Lf)_i = \deg(i) \cdot f_i - \sum_{j \sim i} f_j\), which equals \(\sum_{j \sim i} (f_i - f_j)\). So it measures how much node \(i\)'s value exceeds its neighbors' values:

- **Node 1:** We have \((Lf)_1 = -2\). Node 1 carries value 1 while its sole neighbor, node 2, carries value 3, so node 1 sits below its neighbor in value.
- **Node 2:** Here \((Lf)_2 = 2\). Node 2 has value 3, node 1 has value 1, and node 3 has value 3, so the net imbalance sums to \((3 - 1) + (3 - 3) = 2\).
- **Node 3:** Again \((Lf)_3 = 2\). Node 3 still has value 3 with neighbors 2 and 4 carrying 3 and 1, so \((3 - 3) + (3 - 1) = 2\).
- **Node 4:** Finally \((Lf)_4 = -2\) because node 4 has value 1 but neighbor 3 has value 3, so node 4 sits below its neighbor.

#### (c) Which nodes look "flat" vs. "different"?

No node is perfectly flat, because perfect flatness would mean \((Lf)_i = 0\) for every \(i\). Nodes 2 and 3 behave like peaks in their neighborhoods since \(Lf\) is positive there, while nodes 1 and 4 behave like valleys since \(Lf\) is negative. If \(f\) were constant across the path, every entry of \(Lf\) would be zero, which is how the Laplacian signals that there is no local variation to speak of.

---

### Problem 17. Compare adjacency mixing and Laplacian mixing (10 pts)

Same graph and \(f = (1, 3, 3, 1)^\top\).

#### (a) Compute \(Af\)

\[
(Af)_1 = 0(1) + 1(3) + 0(3) + 0(1) = 3,
\]
\[
(Af)_2 = 1(1) + 0(3) + 1(3) + 0(1) = 1 + 3 = 4,
\]
\[
(Af)_3 = 0(1) + 1(3) + 0(3) + 1(1) = 3 + 1 = 4,
\]
\[
(Af)_4 = 0(1) + 0(3) + 1(3) + 0(1) = 3.
\]

\[
Af = \begin{bmatrix} 3 \\ 4 \\ 4 \\ 3 \end{bmatrix}.
\]

#### (b) Comparison

We have \(Af = (3, 4, 4, 3)^\top\) and \(Lf = (-2, 2, 2, -2)^\top\).

Note that \(Lf = Df - Af\), where \(Df = (1 \cdot 1,\; 2 \cdot 3,\; 2 \cdot 3,\; 1 \cdot 1)^\top = (1, 6, 6, 1)^\top\). Indeed: \((1, 6, 6, 1)^\top - (3, 4, 4, 3)^\top = (-2, 2, 2, -2)^\top\). Check.

#### (c) How \(A\) differs from \(L\) in what they do

Multiplying by \(A\) is a **mixing** operation: each node's new value is the sum of its neighbors' values. It replaces a node's value with an aggregate of its neighborhood, which works like a smoothing/diffusion step.

Multiplying by \(L\) is a **contrast** operation. Each entry measures how much a node's value differs from its neighbors. Large positive values of \((Lf)_i\) mean node \(i\) sits above its neighbors locally, large negative values mean it sits below, and zero means it matches its neighbors exactly.

In GNN language, \(A\) matches the aggregation step that pools neighbor information, while \(L\) matches the idea of measuring local smoothness or disagreement. Spectral GNN filters are often polynomials in \(L\) or in a normalized Laplacian, and those polynomials mix smoothing and sharpening effects in one formula.

---

### Problem 18. A first spectral filter (10 pts)

Define:

\[
g = (I - \alpha L) f, \quad \alpha = \tfrac{1}{2}.
\]

#### (a) Compute \(g\)

From Problem 16, \(Lf = (-2, 2, 2, -2)^\top\). So:

\[
\alpha Lf = \tfrac{1}{2} \begin{bmatrix} -2 \\ 2 \\ 2 \\ -2 \end{bmatrix} = \begin{bmatrix} -1 \\ 1 \\ 1 \\ -1 \end{bmatrix}.
\]

\[
g = f - \alpha Lf = \begin{bmatrix} 1 \\ 3 \\ 3 \\ 1 \end{bmatrix} - \begin{bmatrix} -1 \\ 1 \\ 1 \\ -1 \end{bmatrix} = \begin{bmatrix} 2 \\ 2 \\ 2 \\ 2 \end{bmatrix}.
\]

#### (b) More similar or less similar?

The original \(f = (1, 3, 3, 1)^\top\) has a spread of values (range 1 to 3). After filtering, \(g = (2, 2, 2, 2)^\top\), and all values are identical. The filter made neighboring values **more similar**; in fact, it produced a perfectly smooth (constant) signal in this case.

#### (c) Why this is useful in graph learning

This kind of smoothing is useful because many real graphs exhibit homophily, meaning linked nodes tend to share similar labels or features. A smoothing filter damps noise and pushes neighboring features toward each other, which tends to help node classification and similar tasks. You can view it as regularization that uses the edges to carry inductive bias.

#### (d) Connection with message passing

The filter \(g = (I - \alpha L)f = f - \alpha(Df - Af) = (1 - \alpha D)f + \alpha Af\) can be rewritten as a weighted combination of a node's own feature and the sum of its neighbors' features. That is exactly what a single message-passing layer does: combine self-information with aggregated neighbor information. So spectral filtering with a first-order polynomial in \(L\) matches one round of linear message passing. That identity is what people mean when they relate spectral and spatial pictures of graph networks.

---

## Part V. Learning and Generalization (Conceptual, Ungraded)

---

### Problem 20. Population risk versus empirical risk

#### (a) Difference

Empirical risk is the average loss computed on the training set:

\[
\hat{R}(F) = \frac{1}{N} \sum_{i=1}^{N} \ell(G_i, y_i, F(G_i)).
\]

Population risk is the expected loss over the true data distribution:

\[
R(F) = \mathbb{E}_{(G, y)} \bigl[\ell(G, y, F(G))\bigr].
\]

Empirical risk is what we can compute; population risk is what we actually care about. The difference between them is the generalization gap.

#### (b) Why a model can do well on training graphs but fail on new ones

This situation is what we call overfitting. If the model memorizes quirks of the training graphs such as rare substructures, particular sizes, or odd label mixes instead of learning rules that transfer, it will show low empirical risk but high population risk. The gap between those two risks widens when the model is too flexible relative to how much independent data we actually have.

#### (c) Why this is especially interesting for graphs of different sizes

Graphs come in varying sizes, and a model trained on small instances might later face patterns that simply never appeared in training once graphs grow. The combinatorial space of possible graphs is enormous, so any finite training set leaves huge holes. Node-level models sometimes still generalize when local statistics stay stable, but graph-level models must also cope with global quantities such as diameter, density, and component layout, all of which can shift sharply when \(|V|\) grows.

---

### Problem 21. Extrapolation and graph size

#### (a) Why testing on larger graphs is difficult

The model has never seen the kinds of structural patterns that only emerge at scale, such as long-range dependencies, high-diameter paths, dense subgraphs of sizes not present in training. The statistics of the learned message-passing features may shift when degree distributions or neighborhood densities change. Lecture 16 discusses this directly in the context of physical simulation, where models trained on small particle counts must generalize to 10x more particles.

#### (b) Why local neighborhood patterns matter

If the local structure around each node looks similar in small and large graphs (same degree distribution, same motifs), then a well-trained GNN can generalize, because it fundamentally operates on local patches. This is the locality inductive bias that helps GNNs transfer across sizes.

#### (c) A reason the model might fail despite low training error

If the label truly depends on a global property such as connectivity or the existence of a bottleneck, a model that nails small training graphs might have learned brittle shortcuts such as assuming every small graph it saw was connected. Those shortcuts fail once the test distribution includes larger or differently structured graphs. Low training error therefore only certifies fit on the training distribution, not extrapolation beyond it.

---

### Problem 22. Algorithmic alignment

#### (a) What does "aligned" mean?

I would say an architecture is aligned with a task when the sequence of operations the network performs resembles the sequence of operations in a classical algorithm for that task, so the inductive bias of the network lines up with the problem structure.

#### (b) Why message passing is a good fit for shortest paths

The Bellman-Ford shortest-path algorithm works by iteratively relaxing distances: at each step, each node updates its tentative distance based on its neighbors' distances. This is exactly what message passing does: each node aggregates information from neighbors and updates itself. So the GNN's computation mirrors the algorithm step-for-step, making it easy for the model to learn the right update rule.

#### (c) Another task for which message passing is natural

**Label propagation and community detection** fit the same picture, because updating a node's label from its neighbors is literally one step of message passing. Diffusion models on graphs such as heat flow or PageRank also update each node from its neighbors in discrete iterations, so they slot naturally into the MPNN viewpoint.

---

## Part VI. Reflection (10 pts)

Working through the toy GNN updates and the Weisfeiler-Leman procedure side by side finally made their relationship feel concrete instead of abstract. On the four-vertex path, I watched 1-WL split the endpoints from the interior, and the manual message-passing rule produced the exact same grouping once neighbor sums propagated, which was satisfying to see in numbers rather than on a slide. The case that stuck with me is \(C_6\) versus two disjoint triangles: the graphs are nothing alike globally, yet every vertex shares the same degree-two pattern locally, so neither 1-WL nor the additive toy update can separate them. That was the moment I really believed the slogan that local rules miss global wiring. The Laplacian block complemented that lesson because \(A\) blends neighbor values the way an aggregation step does, whereas \(L\) highlights mismatch along edges, and applying \(I - \alpha L\) bluntly smooths the signal the same way a single linear graph filter would. Lecture 15 also drove home that choices such as summing versus averaging neighbors, adding residuals, or inserting nonlinearities are not cosmetic, because they change which functions the stack can realize. By the end I stopped picturing GNNs as generic black boxes and started picturing them as short programs built from repeated local instructions, which is exactly how I want to think about them on exams.
