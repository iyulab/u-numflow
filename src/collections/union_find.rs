//! Disjoint-set (Union-Find) data structure.
//!
//! Maintains a collection of disjoint sets over elements `0..n` with
//! near-constant-time union and find operations.
//!
//! # Algorithm
//!
//! Uses **path compression** during `find` and **union by rank** during
//! `union` to achieve amortized O(α(n)) per operation, where α is the
//! inverse Ackermann function.
//!
//! For all practical input sizes (n < 2^65536), α(n) ≤ 4, so operations
//! are effectively O(1).
//!
//! # References
//!
//! - Tarjan (1975), "Efficiency of a Good but Not Linear Set Union Algorithm"
//! - Tarjan & van Leeuwen (1984), "Worst-Case Analysis of Set Union Algorithms"

/// Disjoint-set forest with path compression and union by rank.
///
/// # Examples
/// ```
/// use u_optim::collections::UnionFind;
///
/// let mut uf = UnionFind::new(5);
/// assert_eq!(uf.component_count(), 5);
///
/// uf.union(0, 1);
/// uf.union(2, 3);
/// assert_eq!(uf.component_count(), 3);
///
/// assert!(uf.connected(0, 1));
/// assert!(!uf.connected(0, 2));
///
/// uf.union(1, 3);
/// assert!(uf.connected(0, 2)); // transitivity
/// assert_eq!(uf.component_count(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
    size: Vec<usize>,
    components: usize,
}

impl UnionFind {
    /// Creates a new Union-Find with `n` disjoint singleton sets `{0}, {1}, ..., {n-1}`.
    ///
    /// # Complexity
    /// O(n)
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            size: vec![1; n],
            components: n,
        }
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    /// Returns `true` if there are no elements.
    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }

    /// Finds the representative (root) of the set containing `x`.
    ///
    /// Applies **path compression**: every node on the path from `x` to
    /// the root is made a direct child of the root.
    ///
    /// # Complexity
    /// Amortized O(α(n))
    ///
    /// # Panics
    /// Panics if `x >= len()`.
    pub fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    /// Merges the sets containing `x` and `y`.
    ///
    /// Uses **union by rank**: the tree with smaller rank is attached
    /// under the root of the tree with larger rank.
    ///
    /// # Returns
    /// `true` if `x` and `y` were in different sets (and are now merged),
    /// `false` if they were already in the same set.
    ///
    /// # Complexity
    /// Amortized O(α(n))
    ///
    /// # Panics
    /// Panics if `x >= len()` or `y >= len()`.
    pub fn union(&mut self, x: usize, y: usize) -> bool {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x == root_y {
            return false;
        }

        // Union by rank
        match self.rank[root_x].cmp(&self.rank[root_y]) {
            std::cmp::Ordering::Less => {
                self.parent[root_x] = root_y;
                self.size[root_y] += self.size[root_x];
            }
            std::cmp::Ordering::Greater => {
                self.parent[root_y] = root_x;
                self.size[root_x] += self.size[root_y];
            }
            std::cmp::Ordering::Equal => {
                self.parent[root_y] = root_x;
                self.size[root_x] += self.size[root_y];
                self.rank[root_x] += 1;
            }
        }

        self.components -= 1;
        true
    }

    /// Returns `true` if `x` and `y` are in the same set.
    ///
    /// # Complexity
    /// Amortized O(α(n))
    pub fn connected(&mut self, x: usize, y: usize) -> bool {
        self.find(x) == self.find(y)
    }

    /// Returns the number of disjoint sets.
    ///
    /// # Complexity
    /// O(1)
    pub fn component_count(&self) -> usize {
        self.components
    }

    /// Returns the size of the set containing `x`.
    ///
    /// # Complexity
    /// Amortized O(α(n))
    pub fn component_size(&mut self, x: usize) -> usize {
        let root = self.find(x);
        self.size[root]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let uf = UnionFind::new(5);
        assert_eq!(uf.len(), 5);
        assert_eq!(uf.component_count(), 5);
    }

    #[test]
    fn test_new_empty() {
        let uf = UnionFind::new(0);
        assert_eq!(uf.len(), 0);
        assert!(uf.is_empty());
        assert_eq!(uf.component_count(), 0);
    }

    #[test]
    fn test_find_initial() {
        let mut uf = UnionFind::new(5);
        for i in 0..5 {
            assert_eq!(uf.find(i), i);
        }
    }

    #[test]
    fn test_union_basic() {
        let mut uf = UnionFind::new(5);
        assert!(uf.union(0, 1));
        assert!(uf.connected(0, 1));
        assert_eq!(uf.component_count(), 4);
    }

    #[test]
    fn test_union_same_set() {
        let mut uf = UnionFind::new(5);
        uf.union(0, 1);
        assert!(!uf.union(0, 1)); // already same set
        assert_eq!(uf.component_count(), 4);
    }

    #[test]
    fn test_transitivity() {
        let mut uf = UnionFind::new(5);
        uf.union(0, 1);
        uf.union(1, 2);
        assert!(uf.connected(0, 2));
    }

    #[test]
    fn test_not_connected() {
        let mut uf = UnionFind::new(5);
        uf.union(0, 1);
        uf.union(2, 3);
        assert!(!uf.connected(0, 2));
        assert!(!uf.connected(1, 3));
    }

    #[test]
    fn test_merge_components() {
        let mut uf = UnionFind::new(5);
        uf.union(0, 1);
        uf.union(2, 3);
        assert_eq!(uf.component_count(), 3);

        uf.union(1, 3); // merge two components
        assert_eq!(uf.component_count(), 2);
        assert!(uf.connected(0, 2));
        assert!(uf.connected(0, 3));
    }

    #[test]
    fn test_component_size() {
        let mut uf = UnionFind::new(5);
        assert_eq!(uf.component_size(0), 1);

        uf.union(0, 1);
        assert_eq!(uf.component_size(0), 2);
        assert_eq!(uf.component_size(1), 2);

        uf.union(0, 2);
        assert_eq!(uf.component_size(0), 3);
        assert_eq!(uf.component_size(2), 3);
    }

    #[test]
    fn test_all_in_one() {
        let mut uf = UnionFind::new(5);
        for i in 0..4 {
            uf.union(i, i + 1);
        }
        assert_eq!(uf.component_count(), 1);
        assert_eq!(uf.component_size(0), 5);
        for i in 0..5 {
            for j in 0..5 {
                assert!(uf.connected(i, j));
            }
        }
    }

    #[test]
    fn test_single_element() {
        let mut uf = UnionFind::new(1);
        assert_eq!(uf.find(0), 0);
        assert_eq!(uf.component_count(), 1);
        assert_eq!(uf.component_size(0), 1);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(300))]

        #[test]
        fn union_find_transitivity(
            n in 2_usize..20,
            ops in proptest::collection::vec((0_usize..20, 0_usize..20), 0..50),
        ) {
            let mut uf = UnionFind::new(n);
            for &(x, y) in &ops {
                if x < n && y < n {
                    uf.union(x, y);
                }
            }

            // Verify transitivity
            for x in 0..n {
                for y in 0..n {
                    for z in 0..n {
                        if uf.connected(x, y) && uf.connected(y, z) {
                            prop_assert!(
                                uf.connected(x, z),
                                "transitivity violated: {x}~{y} and {y}~{z} but not {x}~{z}"
                            );
                        }
                    }
                }
            }
        }

        #[test]
        fn component_count_invariant(
            n in 1_usize..20,
            ops in proptest::collection::vec((0_usize..20, 0_usize..20), 0..50),
        ) {
            let mut uf = UnionFind::new(n);
            let mut expected_components = n;

            for &(x, y) in &ops {
                if x < n && y < n {
                    let merged = uf.union(x, y);
                    if merged {
                        expected_components -= 1;
                    }
                }
            }

            prop_assert_eq!(uf.component_count(), expected_components);
        }

        #[test]
        fn component_sizes_sum_to_n(
            n in 1_usize..20,
            ops in proptest::collection::vec((0_usize..20, 0_usize..20), 0..30),
        ) {
            let mut uf = UnionFind::new(n);
            for &(x, y) in &ops {
                if x < n && y < n {
                    uf.union(x, y);
                }
            }

            // Sum of sizes of all roots should equal n
            let mut total = 0;
            for i in 0..n {
                if uf.find(i) == i {
                    total += uf.component_size(i);
                }
            }
            prop_assert_eq!(total, n, "component sizes should sum to n");
        }
    }
}
