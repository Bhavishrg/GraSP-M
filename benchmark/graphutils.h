#ifndef GRAPH_GEN_H
#define GRAPH_GEN_H

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <random>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../src/utils/types.h"

using namespace std;
using common::utils::Ring;
using common::utils::wire_t;

// Struct to represent a single daglist entry (vertex or edge)
struct DagEntry {
    Ring src;   // Source vertex ID
    Ring dst;   // Destination vertex ID
    Ring isV;   // 1 if vertex entry, 0 if edge entry
    Ring data;  // Data payload
    Ring sigs;  // Position in source-ordered index
    Ring sigv;  // Position in vertex-ordered index
    Ring sigd;  // Position in destination-ordered index
    
    // Constructor
    DagEntry(Ring src_ = 0, Ring dst_ = 0, Ring isV_ = 0, Ring data_ = 0,
             Ring sigs_ = 0, Ring sigv_ = 0, Ring sigd_ = 0)
        : src(src_), dst(dst_), isV(isV_), data(data_), 
          sigs(sigs_), sigv(sigv_), sigd(sigd_) {}
    
    // Convert to vector<Ring> format for compatibility
    vector<Ring> toVector() const {
        return {src, dst, isV, data, sigs, sigv, sigd};
    }
    
    // Create from vector<Ring>
    static DagEntry fromVector(const vector<Ring>& v) {
        if (v.size() < 7) return DagEntry();
        return DagEntry(v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
    }
};

// Struct to represent a complete daglist graph
struct Daglist {
    vector<DagEntry> entries;
    Ring nV;  // Number of vertices
    Ring nE;  // Number of edges
    
    Daglist() : nV(0), nE(0) {}
    
    Daglist(const vector<DagEntry>& entries_) : entries(entries_) {
        nV = 0;
        nE = 0;
        for (const auto& e : entries) {
            if (e.isV == 1) nV++;
            else nE++;
        }
    }
    
    size_t size() const { return entries.size(); }
    bool empty() const { return entries.empty(); }
    
    // Convert to vector<vector<Ring>> format for compatibility
    vector<vector<Ring>> toVectorFormat() const {
        vector<vector<Ring>> result;
        result.reserve(entries.size());
        for (const auto& e : entries) {
            result.push_back(e.toVector());
        }
        return result;
    }
    
    // Create from vector<vector<Ring>>
    static Daglist fromVectorFormat(const vector<vector<Ring>>& vecs) {
        vector<DagEntry> entries;
        entries.reserve(vecs.size());
        for (const auto& v : vecs) {
            entries.push_back(DagEntry::fromVector(v));
        }
        return Daglist(entries);
    }
};

// Struct to represent distributed daglist across multiple parties
struct DistributedDaglist {
    int num_clients;
    int nV;  // Total number of vertices
    int nE;  // Total number of edges

    vector<DagEntry> VertexList;  // All vertices across parties
    vector<DagEntry> EdgeList;    // All edges across parties

    vector<Ring> SubgraphSizes; // Size of Subgraph per party
    vector<vector<DagEntry>> SubgVertexLists;  // Subgraph corresponding to each party    
    vector<vector<DagEntry>> SubgEdgeLists;    // Subgraph corresponding to each party
    
    // Permutation vectors for shuffling

    vector<vector<Ring>> sigg; // permutation to reorder VertexList such that vertices owned by party i and their immediate neighbors are grouped together
    
    vector<vector<Ring>> sigs; // permutation of size SubgraphSizes[i] to reorder subgraph Subgraphs[i] from vertex order to source order 
    vector<vector<Ring>> sigd; // permutation of size SubgraphSizes[i] to reorder subgraph Subgraphs[i] from source order to destination order
    vector<vector<Ring>> sigv; // permutation of size SubgraphSizes[i] to reorder subgraph Subgraphs[i] from destination order to vertex order
    
    DistributedDaglist() : num_clients(0), nV(0), nE(0) {}
    
    DistributedDaglist(int np) : num_clients(np), nV(0), nE(0) {
        SubgVertexLists.resize(np);
        SubgEdgeLists.resize(np);
        SubgraphSizes.resize(np, 0);
        sigg.resize(np);
        sigs.resize(np);
        sigd.resize(np);
        sigv.resize(np);
    }
  
    
    // Get all entries (vertices + edges) for a specific party
    vector<DagEntry> getPartyEntries(int party_id) const {
        vector<DagEntry> result;
        result.reserve(SubgraphSizes[party_id]);
        result.insert(result.end(), SubgVertexLists[party_id].begin(), SubgVertexLists[party_id].end());
        result.insert(result.end(), SubgEdgeLists[party_id].begin(), SubgEdgeLists[party_id].end());
        return result;
    }
    
    // Get total entries across all parties
    size_t totalSize() const {
        return nV + nE;
    }
};


static inline Ring pack_pair(Ring a, Ring b) {
  return (a << 32) | b;
}

inline vector<pair<Ring, Ring>> generate_scale_free(Ring nV, Ring nE, Ring fixed_seed = 42) {
  vector<pair<Ring, Ring>> edges;
  if (nV == 0 || nE == 0) return edges;

  // degree list for preferential attachment (destination-preferential)
  vector<Ring> degreeList;
  degreeList.reserve(nV * 4 + 10);

  // initialize degreeList with each node once to avoid zero-probability
  for (Ring i = 0; i < nV; ++i) degreeList.push_back(i);

  unordered_set<Ring> seen;
  seen.reserve(nE * 2 + 10);
  mutex edgeMutex, degreeMutex;

  Ring seed = fixed_seed;

  #pragma omp parallel
  {
    // thread-local random engine
    int tid = 0;
    #ifdef _OPENMP
    tid = omp_get_thread_num();
    #endif
    std::mt19937_64 rng(seed + tid);
    uniform_int_distribution<Ring> uniV(0, nV - 1);

    while (true) {
      Ring currentSize;
      {
        lock_guard<mutex> lock(edgeMutex);
        currentSize = edges.size();
        if (currentSize >= nE) break;
      }

      Ring src = uniV(rng);
      Ring dst;
      
      {
        lock_guard<mutex> lock(degreeMutex);
        if (degreeList.empty()) break;
        dst = degreeList[rng() % degreeList.size()];
      }

      if (src == dst) continue; // avoid self-loops

      Ring key = pack_pair(src, dst);
      
      {
        lock_guard<mutex> lock(edgeMutex);
        if (edges.size() >= nE) break;
        if (seen.find(key) != seen.end()) continue; // avoid duplicate

        edges.emplace_back(src, dst);
        seen.insert(key);
      }

      // update degreeList to increase attachment probability
      {
        lock_guard<mutex> lock(degreeMutex);
        degreeList.push_back(dst);
        degreeList.push_back(src);
      }
    }
  }

  return edges;
}

inline Daglist build_daglist(Ring nV, const vector<pair<Ring, Ring>>& edges) {
  Ring nE = edges.size();
  Ring total = nV + nE;

  // Build in vertex order: vertices, then edges grouped by source
  vector<DagEntry> entries;
  entries.reserve(total);

  // First, add all vertex entries
  for (Ring v = 0; v < nV; ++v) {
    entries.emplace_back(v, v, 1, 0, 0, 0, 0);
  }

  // Group edges by source vertex
  vector<vector<Ring>> outEdges(nV);
  for (Ring i = 0; i < nE; ++i) {
    outEdges[edges[i].first].push_back(i);
  }

  // Add edges grouped by source
  for (Ring v = 0; v < nV; ++v) {
    for (Ring edgeIdx : outEdges[v]) {
      entries.emplace_back(edges[edgeIdx].first, edges[edgeIdx].second, 0, 0, 0, 0, 0);
    }
  }

  // Build source-ordered index: for each vertex, vertex entry then its out-edges
  vector<Ring> srcOrder; srcOrder.reserve(total);
  for (Ring v = 0; v < nV; ++v) {
    srcOrder.push_back(v);
    Ring edgeStart = nV;
    for (Ring u = 0; u < v; ++u) {
      edgeStart += outEdges[u].size();
    }
    for (Ring i = 0; i < outEdges[v].size(); ++i) {
      srcOrder.push_back(edgeStart + i);
    }
  }

  // Build dest-ordered index: for each vertex, its in-edges then the vertex entry
  vector<vector<Ring>> inEdges(nV);
  Ring edgePos = nV;
  for (Ring v = 0; v < nV; ++v) {
    for (Ring i = 0; i < outEdges[v].size(); ++i) {
      Ring dst = entries[edgePos].dst;
      inEdges[dst].push_back(edgePos);
      edgePos++;
    }
  }

  vector<Ring> dstOrder; dstOrder.reserve(total);
  for (Ring v = 0; v < nV; ++v) {
    for (Ring edgeIdx : inEdges[v]) {
      dstOrder.push_back(edgeIdx);
    }
    dstOrder.push_back(v);
  }

  // vertex-order is the current layout (all vertices, then edges grouped by source)
  // Build reverse maps: position of each index in each ordering
  vector<Ring> pos_in_src(total, 0), pos_in_dst(total, 0), pos_in_vert(total, 0);
  
  #pragma omp parallel for
  for (Ring i = 0; i < srcOrder.size(); ++i) pos_in_src[srcOrder[i]] = i;
  
  #pragma omp parallel for
  for (Ring i = 0; i < dstOrder.size(); ++i) pos_in_dst[dstOrder[i]] = i;
  
  #pragma omp parallel for
  for (Ring i = 0; i < total; ++i) pos_in_vert[i] = i;

  // Fill sigs, sigv, sigd fields
  #pragma omp parallel for
  for (Ring idx = 0; idx < total; ++idx) {
    entries[idx].sigs = pos_in_src[idx];
    entries[idx].sigv = pos_in_vert[idx];
    entries[idx].sigd = pos_in_dst[idx];
    if (idx == 0) {
      entries[idx].data = 1; // Vertex data initialized to 1 for first vertex
    }
  }

  return Daglist(entries);
}

inline DistributedDaglist distribute_daglist(const Daglist& daglist, int num_clients) {
  DistributedDaglist result(num_clients);
  result.nV = daglist.nV;
  result.nE = daglist.nE;
  
  Ring nV = daglist.nV;
  Ring nE = daglist.nE;
  
  if (nV == 0 || num_clients == 0) return result;
  
  // Separate vertices and edges from daglist
  result.VertexList.reserve(nV);
  result.EdgeList.reserve(nE);
  
  for (const auto& entry : daglist.entries) {
    if (entry.isV == 1) {
      result.VertexList.push_back(entry);
    } else {
      result.EdgeList.push_back(entry);
    }
  }
  
  // Step 1: Determine vertex ownership ranges
  Ring vertices_per_client = nV / num_clients;
  Ring remainder = nV % num_clients;
  
  vector<Ring> vertex_start(num_clients);
  vector<Ring> vertex_end(num_clients);
  Ring current_start = 0;
  for (int i = 0; i < num_clients; ++i) {
    vertex_start[i] = current_start;
    Ring count = vertices_per_client + (i < remainder ? 1 : 0);
    vertex_end[i] = current_start + count;
    current_start = vertex_end[i];
  }
  
  // Build edge index by destination
  vector<vector<Ring>> edges_by_dest(nV);
  for (Ring i = 0; i < result.EdgeList.size(); ++i) {
    Ring dst = result.EdgeList[i].dst;
    if (dst < nV) {
      edges_by_dest[dst].push_back(i);
    }
  }
  
  // For each client
  for (int client = 0; client < num_clients; ++client) {
    // Step 2: Build sigg[client] - permutation (position map) that places owned vertices first
    // ordered_vertices: list of vertex indices in new order
    vector<Ring> ordered_vertices;
    ordered_vertices.reserve(nV);
    // owned vertices first
    for (Ring v = vertex_start[client]; v < vertex_end[client]; ++v) ordered_vertices.push_back(v);
    // then the rest in original order
    for (Ring v = 0; v < vertex_start[client]; ++v) ordered_vertices.push_back(v);
    for (Ring v = vertex_end[client]; v < nV; ++v) ordered_vertices.push_back(v);
    
    // build position map: for each original vertex index, position in reordered list
    result.sigg[client].assign(nV, (Ring)0);
    for (Ring pos = 0; pos < (Ring)ordered_vertices.size(); ++pos) {
      result.sigg[client][ordered_vertices[pos]] = pos;
    }
    
    // Step 3: Build SubgEdgeLists[client] - edges with dst at owned vertices
    result.SubgEdgeLists[client].clear();
    for (Ring v = vertex_start[client]; v < vertex_end[client]; ++v) {
      for (Ring edge_idx : edges_by_dest[v]) {
        result.SubgEdgeLists[client].push_back(result.EdgeList[edge_idx]);
      }
    }
    
    // Step 4: Build SubgVertexLists[client] - first min(2*|SubgEdges|, |VertexList|) entries of reordered VertexList
    Ring num_subg_edges = result.SubgEdgeLists[client].size();
    Ring vertex_cap = min(2 * num_subg_edges, (Ring)result.VertexList.size());
    
    result.SubgVertexLists[client].clear();
    result.SubgVertexLists[client].reserve(vertex_cap);
    // ordered_vertices gives vertex indices in the new order; take first vertex_cap of them
    for (Ring i = 0; i < vertex_cap; ++i) {
      Ring v_idx = ordered_vertices[i];
      result.SubgVertexLists[client].push_back(result.VertexList[v_idx]);
    }
    
    // Subgraph size
    Ring subgraph_size = vertex_cap + num_subg_edges;
    result.SubgraphSizes[client] = subgraph_size;
    
    // Build combined subgraph: SubgVertexLists[client] || SubgEdgeLists[client]
    vector<DagEntry> subgraph;
    subgraph.reserve(subgraph_size);
    subgraph.insert(subgraph.end(), result.SubgVertexLists[client].begin(), result.SubgVertexLists[client].end());
    subgraph.insert(subgraph.end(), result.SubgEdgeLists[client].begin(), result.SubgEdgeLists[client].end());
    
    // Step 5: Build sigs[client] - permutation from vertex order to source order
    // Source order: for each vertex, vertex entry followed by its outgoing edges
    
    // Build mapping: which edges have which source (within subgraph)
    unordered_map<Ring, vector<Ring>> edges_by_source;
    for (Ring i = 0; i < num_subg_edges; ++i) {
      Ring src = result.SubgEdgeLists[client][i].src;
      edges_by_source[src].push_back(vertex_cap + i); // Position in combined subgraph
    }
    
    vector<Ring> source_order;
    source_order.reserve(subgraph_size);
    
    // For each vertex in SubgVertexLists (in order)
    for (Ring i = 0; i < vertex_cap; ++i) {
      Ring v = subgraph[i].src; // Vertex ID (src == dst for vertex entries)
      source_order.push_back(i); // Add vertex entry
      
      // Add its outgoing edges that are in SubgEdgeLists
      if (edges_by_source.count(v)) {
        for (Ring edge_pos : edges_by_source[v]) {
          source_order.push_back(edge_pos);
        }
      }
    }
    
    // sigs: position map from vertex-order index (combined index) -> position in source order
    result.sigs[client].assign(subgraph_size, (Ring)0);
    for (Ring i = 0; i < (Ring)source_order.size(); ++i) {
      result.sigs[client][source_order[i]] = i;
    }
    
    // Step 6: Build sigd[client] - permutation from source order to destination order
    // Destination order: for each vertex, incoming edges followed by vertex entry
    
    // Build mapping: which edges have which destination (within subgraph)
    unordered_map<Ring, vector<Ring>> edges_by_dest_sub;
    for (Ring i = 0; i < num_subg_edges; ++i) {
      Ring dst = result.SubgEdgeLists[client][i].dst;
      edges_by_dest_sub[dst].push_back(vertex_cap + i);
    }
    
    vector<Ring> dest_order;
    dest_order.reserve(subgraph_size);
    
    // For each vertex in SubgVertexLists (in order)
    for (Ring i = 0; i < vertex_cap; ++i) {
      Ring v = subgraph[i].dst; // Vertex ID
      
      // Add incoming edges first
      if (edges_by_dest_sub.count(v)) {
        for (Ring edge_pos : edges_by_dest_sub[v]) {
          dest_order.push_back(edge_pos);
        }
      }
      
      // Then add vertex entry
      dest_order.push_back(i);
    }
    
    // Build reverse mapping: combined index -> position in dest order
    vector<Ring> pos_in_dest(subgraph_size, (Ring)0);
    for (Ring i = 0; i < (Ring)dest_order.size(); ++i) {
      pos_in_dest[dest_order[i]] = i;
    }
    
    // sigd: map from source-order position -> dest-order position
    result.sigd[client].assign(subgraph_size, (Ring)0);
    for (Ring i = 0; i < (Ring)source_order.size(); ++i) {
      Ring item = source_order[i]; // combined index
      result.sigd[client][i] = pos_in_dest[item];
    }
    
    // Step 7: Build sigv[client] - permutation from destination order to vertex order
    // Vertex order is the original: SubgVertexLists || SubgEdgeLists (indices 0 to subgraph_size-1)
    
    // Build reverse mapping: combined index -> position in vertex-order (identity: 0..subgraph_size-1)
    vector<Ring> pos_in_vertex(subgraph_size);
    for (Ring i = 0; i < subgraph_size; ++i) pos_in_vertex[i] = i;
    
    // sigv: map from dest-order position -> vertex-order position
    result.sigv[client].assign(subgraph_size, (Ring)0);
    for (Ring i = 0; i < (Ring)dest_order.size(); ++i) {
      Ring item = dest_order[i]; // combined index
      result.sigv[client][i] = pos_in_vertex[item];
    }
  }
  
  return result;
}


// Create circuit inputs from a Daglist
// Populates the inputs map with values from the daglist entries based on wire assignments
// wire_idx tracks the current position in input_wires
inline void set_daglist_inputs(
    const Daglist& daglist,
    const std::vector<wire_t>& input_wires,
    std::unordered_map<wire_t, Ring>& inputs,
    size_t& wire_idx) {
  
  size_t vec_size = daglist.size();
  
  // Initialize all daglist field values
  std::vector<Ring> src_values(vec_size);
  std::vector<Ring> dst_values(vec_size);
  std::vector<Ring> isV_values(vec_size);
  std::vector<Ring> data_values(vec_size);
  std::vector<Ring> sigs_values(vec_size);
  std::vector<Ring> sigv_values(vec_size);
  std::vector<Ring> sigd_values(vec_size);
  
  for (size_t i = 0; i < vec_size; ++i) {
    src_values[i] = daglist.entries[i].src;
    dst_values[i] = daglist.entries[i].dst;
    isV_values[i] = daglist.entries[i].isV;
    data_values[i] = daglist.entries[i].data;
    sigs_values[i] = daglist.entries[i].sigs;
    sigv_values[i] = daglist.entries[i].sigv;
    sigd_values[i] = daglist.entries[i].sigd;
  }
  
  // Set all daglist fields as inputs (7 * vec_size inputs total)
  // The order must match the circuit wire creation order
  
  // src inputs
  for (size_t i = 0; i < vec_size && wire_idx < input_wires.size(); ++i, ++wire_idx) {
    inputs[input_wires[wire_idx]] = src_values[i];
  }
  
  // dst inputs
  for (size_t i = 0; i < vec_size && wire_idx < input_wires.size(); ++i, ++wire_idx) {
    inputs[input_wires[wire_idx]] = dst_values[i];
  }
  
  // isV inputs
  for (size_t i = 0; i < vec_size && wire_idx < input_wires.size(); ++i, ++wire_idx) {
    inputs[input_wires[wire_idx]] = isV_values[i];
  }
  
  // data inputs
  for (size_t i = 0; i < vec_size && wire_idx < input_wires.size(); ++i, ++wire_idx) {
    inputs[input_wires[wire_idx]] = data_values[i];
  }
  
  // sigs inputs (position map for propagate)
  for (size_t i = 0; i < vec_size && wire_idx < input_wires.size(); ++i, ++wire_idx) {
    inputs[input_wires[wire_idx]] = sigs_values[i];
  }
  
  // sigv inputs (position map for gather)
  for (size_t i = 0; i < vec_size && wire_idx < input_wires.size(); ++i, ++wire_idx) {
    inputs[input_wires[wire_idx]] = sigv_values[i];
  }
  
  // sigd inputs (position map for permlist)
  for (size_t i = 0; i < vec_size && wire_idx < input_wires.size(); ++i, ++wire_idx) {
    inputs[input_wires[wire_idx]] = sigd_values[i];
  }
}

// Create circuit inputs from a Daglist (overload with initial wire_idx = 0)
inline void set_daglist_inputs(
    const Daglist& daglist,
    const std::vector<wire_t>& input_wires,
    std::unordered_map<wire_t, Ring>& inputs) {
  size_t wire_idx = 0;
  set_daglist_inputs(daglist, input_wires, inputs, wire_idx);
}





#endif // GRAPH_GEN_H
