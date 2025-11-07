#ifndef GRAPH_GEN_H
#define GRAPH_GEN_H

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <random>
#include <unordered_set>
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
    int num_parties;
    int nV;  // Total number of vertices
    int nE;  // Total number of edges
    vector<vector<DagEntry>> VertexLists;  // VertexLists[i] = vertices owned by party i
    vector<vector<DagEntry>> EdgeLists;    // EdgeLists[i] = edges owned by party i
    vector<Ring> VSizes;  // Number of vertices in each VertexList
    vector<Ring> ESizes;  // Number of edges in each EdgeList
    
    DistributedDaglist() : num_parties(0), nV(0), nE(0) {}
    
    DistributedDaglist(int np) : num_parties(np), nV(0), nE(0) {
        VertexLists.resize(np);
        EdgeLists.resize(np);
        VSizes.resize(np, 0);
        ESizes.resize(np, 0);
    }
    
    DistributedDaglist(int np, int total_nV, int total_nE) 
        : num_parties(np), nV(total_nV), nE(total_nE) {
        VertexLists.resize(np);
        EdgeLists.resize(np);
        VSizes.resize(np, 0);
        ESizes.resize(np, 0);
    }
    
    // Get all entries (vertices + edges) for a specific party
    vector<DagEntry> getPartyEntries(int party_id) const {
        vector<DagEntry> result;
        result.reserve(VSizes[party_id] + ESizes[party_id]);
        result.insert(result.end(), VertexLists[party_id].begin(), VertexLists[party_id].end());
        result.insert(result.end(), EdgeLists[party_id].begin(), EdgeLists[party_id].end());
        return result;
    }
    
    // Get total entries across all parties
    size_t totalSize() const {
        return nV + nE;
    }
};

// Struct to represent a single daglist entry with secret-shared wires (for circuit construction)
struct SSDagEntry {
    wire_t src;   // Wire for source vertex ID
    wire_t dst;   // Wire for destination vertex ID
    wire_t isV;   // Wire for isVertex flag (1 if vertex entry, 0 if edge entry)
    wire_t data;  // Wire for data payload
    wire_t sigs;  // Wire for position in source-ordered index
    wire_t sigv;  // Wire for position in vertex-ordered index
    wire_t sigd;  // Wire for position in destination-ordered index
    
    // Constructor
    SSDagEntry(wire_t src_ = 0, wire_t dst_ = 0, wire_t isV_ = 0, wire_t data_ = 0,
               wire_t sigs_ = 0, wire_t sigv_ = 0, wire_t sigd_ = 0)
        : src(src_), dst(dst_), isV(isV_), data(data_), 
          sigs(sigs_), sigv(sigv_), sigd(sigd_) {}
};

// Struct to represent a complete daglist graph with secret-shared wires
struct SSDaglist {
    vector<SSDagEntry> entries;
    Ring nV;  // Number of vertices (plaintext metadata)
    Ring nE;  // Number of edges (plaintext metadata)
    
    SSDaglist() : nV(0), nE(0) {}
    
    SSDaglist(const vector<SSDagEntry>& entries_, Ring nV_, Ring nE_) 
        : entries(entries_), nV(nV_), nE(nE_) {}
    
    size_t size() const { return entries.size(); }
    bool empty() const { return entries.empty(); }
};

// Struct to represent distributed daglist with secret-shared wires across multiple parties
struct SSDistributedDaglist {
    int num_parties;
    int nV;  // Total number of vertices (plaintext metadata)
    int nE;  // Total number of edges (plaintext metadata)
    vector<vector<SSDagEntry>> VertexLists;  // VertexLists[i] = secret-shared vertices owned by party i
    vector<vector<SSDagEntry>> EdgeLists;    // EdgeLists[i] = secret-shared edges owned by party i
    vector<Ring> VSizes;  // Number of vertices in each VertexList
    vector<Ring> ESizes;  // Number of edges in each EdgeList
    
    SSDistributedDaglist() : num_parties(0), nV(0), nE(0) {}
    
    SSDistributedDaglist(int np) : num_parties(np), nV(0), nE(0) {
        VertexLists.resize(np);
        EdgeLists.resize(np);
        VSizes.resize(np, 0);
        ESizes.resize(np, 0);
    }
    
    SSDistributedDaglist(int np, int total_nV, int total_nE) 
        : num_parties(np), nV(total_nV), nE(total_nE) {
        VertexLists.resize(np);
        EdgeLists.resize(np);
        VSizes.resize(np, 0);
        ESizes.resize(np, 0);
    }
    
    // Get all entries (vertices + edges) for a specific party
    vector<SSDagEntry> getPartyEntries(int party_id) const {
        vector<SSDagEntry> result;
        result.reserve(VSizes[party_id] + ESizes[party_id]);
        result.insert(result.end(), VertexLists[party_id].begin(), VertexLists[party_id].end());
        result.insert(result.end(), EdgeLists[party_id].begin(), EdgeLists[party_id].end());
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

inline vector<pair<Ring, Ring>> generate_scale_free(Ring nV, Ring nE) {
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

  Ring seed = (Ring)chrono::high_resolution_clock::now().time_since_epoch().count();

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

// Distribute daglist_graph across np parties
// Input: daglist_graph (nV vertices + nE edges), np (number of parties)
// Output: DistributedDaglist with np party graphs, where party i owns V/np vertices and their outgoing edges
inline DistributedDaglist distribute_daglist(const Daglist& daglist_graph, int np) {
  if (np <= 0) {
    throw std::invalid_argument("Number of parties must be positive");
  }
  if (daglist_graph.empty()) {
    return DistributedDaglist(np, 0, 0);
  }

  Ring nV = daglist_graph.nV;
  Ring nE = daglist_graph.nE;

  if (nV == 0) {
    return DistributedDaglist(np, 0, nE);
  }

  // Calculate vertices per party
  Ring verts_per_party = (nV + np - 1) / np;  // Ceiling division
  
  // Initialize result
  DistributedDaglist result(np, nV, nE);
  
  // Build a map from vertex ID to party assignment
  // Party i owns vertices [i * verts_per_party, (i+1) * verts_per_party)
  auto get_party = [&](Ring vertex_id) -> int {
    Ring party = vertex_id / verts_per_party;
    return std::min(static_cast<Ring>(party), static_cast<Ring>(np - 1));
  };

  // Distribute entries
  for (const auto& entry : daglist_graph.entries) {
    if (entry.isV == 1) {
      // This is a vertex entry - assign to the party that owns this vertex
      int party = get_party(entry.src);
      result.VertexLists[party].push_back(entry);
      result.VSizes[party]++;
    } else {
      // This is an edge entry - assign to the party that owns the source vertex
      int party = get_party(entry.src);
      result.EdgeLists[party].push_back(entry);
      result.ESizes[party]++;
    }
  }

  return result;
}

// Build a complete Daglist from a DistributedDaglist
// Reconstructs the full graph by merging all party entries
inline Daglist build_daglist_from_distributed(const DistributedDaglist& dist_daglist) {
  vector<DagEntry> all_entries;
  all_entries.reserve(dist_daglist.nV + dist_daglist.nE);
  
  // Collect all vertices first, then all edges (maintaining vertex-order structure)
  for (int i = 0; i < dist_daglist.num_parties; ++i) {
    all_entries.insert(all_entries.end(), 
                      dist_daglist.VertexLists[i].begin(), 
                      dist_daglist.VertexLists[i].end());
  }
  
  for (int i = 0; i < dist_daglist.num_parties; ++i) {
    all_entries.insert(all_entries.end(), 
                      dist_daglist.EdgeLists[i].begin(), 
                      dist_daglist.EdgeLists[i].end());
  }
  
  return Daglist(all_entries);
}

// Build a complete SSDaglist from an SSDistributedDaglist
// Reconstructs the full graph with secret-shared wires by merging all party entries
inline SSDaglist build_ssdaglist_from_distributed(const SSDistributedDaglist& dist_daglist) {
  vector<SSDagEntry> all_entries;
  all_entries.reserve(dist_daglist.nV + dist_daglist.nE);
  
  // Collect all vertices first, then all edges (maintaining vertex-order structure)
  for (int i = 0; i < dist_daglist.num_parties; ++i) {
    all_entries.insert(all_entries.end(), 
                      dist_daglist.VertexLists[i].begin(), 
                      dist_daglist.VertexLists[i].end());
  }
  
  for (int i = 0; i < dist_daglist.num_parties; ++i) {
    all_entries.insert(all_entries.end(), 
                      dist_daglist.EdgeLists[i].begin(), 
                      dist_daglist.EdgeLists[i].end());
  }
  
  return SSDaglist(all_entries, dist_daglist.nV, dist_daglist.nE);
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
