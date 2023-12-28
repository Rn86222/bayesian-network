use std::{collections::HashMap, fmt::Debug, hash::Hash};

type Name = String;
type Probability = f64;
type NodeId = usize;

/// The type of a node in a Bayesian network.
#[derive(Clone)]
pub enum NodeType<T> {
    Root(HashMap<T, Probability>),
    Leaf,
    Intermediate,
}

impl<T: Clone + PartialEq + Eq + Hash + Debug> Debug for NodeType<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeType::Root(prob) => {
                let mut s = String::new();
                for (value, prob) in prob {
                    s.push_str(&format!("  {:?}: {}\n", value, prob));
                }
                write!(f, "Root(\n{})", s)
            }
            NodeType::Leaf => write!(f, "Leaf"),
            NodeType::Intermediate => write!(f, "Intermediate"),
        }
    }
}

struct Node<T> {
    id: NodeId,
    parents: Vec<NodeId>,
    children: Vec<NodeId>,
    probability: HashMap<Vec<T>, HashMap<T, Probability>>,
    node_type: NodeType<T>,
}

impl<T> Node<T> {
    fn new(node_type: NodeType<T>, id: NodeId) -> Node<T> {
        Node {
            id,
            parents: Vec::new(),
            children: Vec::new(),
            probability: HashMap::new(),
            node_type,
        }
    }
}

/// A Bayesian network.
pub struct BayesianNetwork<T: Clone + PartialEq + Eq + Hash + Debug> {
    nodes: Vec<Node<T>>,
    node_map: HashMap<Name, NodeId>,
    value_space: Vec<T>,
}

impl<T: Clone + PartialEq + Eq + Hash + Debug> Debug for BayesianNetwork<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        s.push_str("--------------------\n");
        s.push_str("| Bayesian Network |\n");
        s.push_str("--------------------\n");
        for node in &self.nodes {
            s.push_str(&format!("{}: {:?}\n", node.id, node.node_type));
            match node.node_type {
                NodeType::Root(_) => {
                    s.push_str(&format!("  children: {:?}\n\n", node.children));
                }
                NodeType::Leaf => {
                    s.push_str(&format!("  parents: {:?}\n", node.parents));
                    s.push_str(&format!("  probability: {:?}\n\n", node.probability));
                }
                NodeType::Intermediate => {
                    s.push_str(&format!("  parents: {:?}\n", node.parents));
                    s.push_str(&format!("  probability: {:?}\n", node.probability));
                    s.push_str(&format!("  children: {:?}\n\n", node.children));
                }
            }
        }
        write!(f, "{}", s)
    }
}

impl<T: Clone + PartialEq + Eq + Hash + Debug> BayesianNetwork<T> {
    /// Create a new Bayesian network with the given value space.
    pub fn new(value_space: Vec<T>) -> BayesianNetwork<T> {
        BayesianNetwork {
            nodes: Vec::new(),
            node_map: HashMap::new(),
            value_space,
        }
    }

    /// Add a node to the network.
    pub fn add_node(&mut self, name: &str, node_type: NodeType<T>) {
        if let NodeType::Root(prob) = &node_type {
            for value in &self.value_space {
                if !prob.contains_key(value) {
                    panic!(
                        "Root node `{}` probability map does not contain all values in value space",
                        name
                    );
                }
            }
            let mut sum = 0.0;
            for value in prob.keys() {
                if !self.value_space.contains(value) {
                    panic!(
                        "Root node `{}` probability map contains value not in value space",
                        name
                    );
                }
                sum += prob[value];
            }
            if (sum - 1.0).abs() > 0.0000001 {
                eprintln!(
                    "Warning: Root node `{}` probability map may not sum to 1.0",
                    name
                );
            }
        }
        let id = self.nodes.len();
        self.nodes.push(Node::new(node_type, id));
        self.node_map.insert(name.to_string(), id);
    }

    /// Add a dependency to the network.
    pub fn add_dependency(
        &mut self,
        parent_names: Vec<&str>,
        child_name: &str,
        prob: HashMap<Vec<T>, HashMap<T, Probability>>,
    ) {
        for (key, map) in &prob {
            if key.len() != parent_names.len() {
                panic!("Dependency probability map key length does not match parent length ({} and {}, {:?} -> {})", key.len(), parent_names.len(), parent_names, child_name);
            }
            for value in key {
                if !self.value_space.contains(value) {
                    panic!(
                        "Dependency probability map key contains value not in value space ({:?}, {:?} -> {})",
                        value, parent_names, child_name
                    );
                }
            }
            for value in &self.value_space {
                if !map.contains_key(value) {
                    panic!("Dependency probability map does not contain all values in value space ({:?}, {:?} -> {})", value, parent_names, child_name);
                }
            }
            let mut sum = 0.0;
            for value in map.keys() {
                sum += map[value];
            }
            if (sum - 1.0).abs() > 0.0000001 {
                eprintln!(
                    "Warning: Dependency probability map may not sum to 1.0 ({:?} -> {})",
                    parent_names, child_name
                );
            }
        }
        let child_id = self.node_map[child_name];
        for parent_name in parent_names {
            let parent_id = self.node_map[parent_name];
            if let NodeType::Leaf = self.nodes[parent_id].node_type {
                panic!("Cannot add dependency from leaf node `{}`", parent_name);
            }
            if let NodeType::Root(_) = self.nodes[child_id].node_type {
                panic!("Cannot add dependency to root node `{}`", child_name);
            }
            self.nodes[parent_id].children.push(child_id);
            self.nodes[child_id].parents.push(parent_id);
        }
        self.nodes[child_id].probability = prob;
    }

    fn get_node_index(&self, name: &str) -> NodeId {
        self.node_map[name]
    }

    fn pass_pi(
        &self,
        node: &Node<T>,
        child: &NodeId,
        evidence: &HashMap<NodeId, T>,
        lambda_map: &HashMap<(NodeId, NodeId), HashMap<T, Probability>>,
        pi_map: &mut HashMap<(NodeId, NodeId), HashMap<T, Probability>>,
    ) {
        let mut map = HashMap::new();
        for value in &self.value_space {
            let mut lambda = 1.0;
            for other_child in &node.children {
                if *other_child != *child {
                    lambda *= lambda_map[&(*other_child, node.id)][value];
                }
            }
            let mut sum = 0.0;
            if let Some(evidence_value) = evidence.get(&node.id) {
                sum = if *evidence_value == *value { 1.0 } else { 0.0 };
            } else if let NodeType::Root(prob_map) = &node.node_type {
                sum = prob_map[value];
            } else {
                let parents = &node.parents;
                let evident_parents = parents
                    .iter()
                    .enumerate()
                    .filter(|(_, parent)| evidence.contains_key(parent))
                    .map(|(i, _)| i)
                    .collect::<Vec<usize>>();
                for (parent_values, prob) in &node.probability {
                    let mut skip = false;
                    for e in &evident_parents {
                        if parent_values[*e] != evidence[&parents[*e]] {
                            skip = true;
                            break;
                        }
                    }
                    if skip {
                        continue;
                    }
                    let mut parent_mul = 1.0;
                    for (i, parent) in node.parents.iter().enumerate() {
                        parent_mul *= pi_map[&(*parent, node.id)][&parent_values[i]];
                    }
                    sum += prob[value] * parent_mul;
                }
            }
            map.insert(value.clone(), lambda * sum);
        }
        pi_map.insert((node.id, *child), map);
    }

    fn pass_lambda(
        &self,
        node: &Node<T>,
        parent: &NodeId,
        parent_index: usize,
        evidence: &HashMap<NodeId, T>,
        pi_map: &HashMap<(NodeId, NodeId), HashMap<T, Probability>>,
        lambda_map: &mut HashMap<(NodeId, NodeId), HashMap<T, Probability>>,
    ) {
        let mut map = HashMap::new();
        for value in &self.value_space {
            let mut sum = 0.0;
            let parents = &node.parents;
            let evident_parents = parents
                .iter()
                .enumerate()
                .filter(|(_, parent)| evidence.contains_key(parent))
                .map(|(i, _)| i)
                .collect::<Vec<usize>>();
            for (parent_values, prob) in &node.probability {
                if parent_values[parent_index] != *value {
                    continue;
                }
                let mut skip = false;
                for e in &evident_parents {
                    if parent_values[*e] != evidence[&parents[*e]] {
                        skip = true;
                        break;
                    }
                }
                if skip {
                    continue;
                }
                let mut parent_mul = 1.0;
                for (i, other_parent) in node.parents.iter().enumerate() {
                    if other_parent != parent {
                        parent_mul *= pi_map[&(*other_parent, node.id)][&parent_values[i]];
                    }
                }
                let mut node_sum = 0.0;
                let evid = evidence.get(&node.id);
                for node_value in &self.value_space {
                    if evid.is_some() && evid.unwrap() != node_value {
                        continue;
                    }
                    let mut lambda = 1.0;
                    for child in &node.children {
                        lambda *= lambda_map[&(*child, node.id)][node_value];
                    }
                    node_sum += lambda * prob[node_value];
                }
                sum += parent_mul * node_sum;
            }
            map.insert(value.clone(), sum);
        }
        lambda_map.insert((node.id, *parent), map);
    }

    /// Infer the probability of each node given the evidence.
    pub fn infer(&self, evidence: &HashMap<&str, T>) -> Vec<HashMap<T, Probability>> {
        let mut pi_map = HashMap::new();
        let mut lambda_map = HashMap::new();
        let mut _evidence: HashMap<NodeId, T> = HashMap::new();
        for (name, value) in evidence {
            _evidence.insert(self.node_map[&name.to_string()], value.clone());
        }
        let evidence = &_evidence;

        let mut update = false;
        loop {
            for node in &self.nodes {
                if node
                    .parents
                    .iter()
                    .all(|parent_id| pi_map.contains_key(&(*parent_id, node.id)))
                {
                    let children_num = node.children.len();
                    let mut lambda_count = 0;
                    for child in &node.children {
                        if lambda_map.contains_key(&(*child, node.id)) {
                            lambda_count += 1;
                        }
                    }
                    if lambda_count + 2 <= children_num {
                        continue;
                    } else if lambda_count + 1 == children_num {
                        for child in &node.children {
                            if !lambda_map.contains_key(&(*child, node.id)) {
                                if pi_map.contains_key(&(node.id, *child)) {
                                    continue;
                                }
                                update = true;
                                self.pass_pi(node, child, evidence, &lambda_map, &mut pi_map);
                            }
                        }
                    } else {
                        for child in &node.children {
                            if pi_map.contains_key(&(node.id, *child)) {
                                continue;
                            }
                            update = true;
                            self.pass_pi(node, child, evidence, &lambda_map, &mut pi_map);
                        }
                    }
                }

                if node
                    .children
                    .iter()
                    .all(|child_id| lambda_map.contains_key(&(*child_id, node.id)))
                {
                    let parents_num = node.parents.len();
                    let mut pi_count = 0;
                    for parent in &node.parents {
                        if pi_map.contains_key(&(*parent, node.id)) {
                            pi_count += 1;
                        }
                    }
                    if pi_count + 2 <= parents_num {
                        continue;
                    } else if pi_count + 1 == parents_num {
                        for (parent_index, parent) in node.parents.iter().enumerate() {
                            if !pi_map.contains_key(&(*parent, node.id)) {
                                if lambda_map.contains_key(&(node.id, *parent)) {
                                    continue;
                                }
                                update = true;
                                self.pass_lambda(
                                    node,
                                    parent,
                                    parent_index,
                                    evidence,
                                    &pi_map,
                                    &mut lambda_map,
                                );
                            }
                        }
                    } else {
                        for (parent_index, parent) in node.parents.iter().enumerate() {
                            if lambda_map.contains_key(&(node.id, *parent)) {
                                continue;
                            }
                            update = true;
                            self.pass_lambda(
                                node,
                                parent,
                                parent_index,
                                evidence,
                                &pi_map,
                                &mut lambda_map,
                            );
                        }
                    }
                }
            }
            if !update {
                break;
            } else {
                update = false;
            }
        }
        let mut inferred_probabilities = Vec::new();
        for node in &self.nodes {
            let mut map = HashMap::new();
            let mut probs = Vec::new();
            for value in &self.value_space {
                let mut lambda = 1.0;
                for child in &node.children {
                    lambda *= lambda_map[&(*child, node.id)][value];
                }
                let pi = if evidence.contains_key(&node.id) {
                    if evidence[&node.id] == *value {
                        1.0
                    } else {
                        0.0
                    }
                } else if let NodeType::Root(prob_map) = &node.node_type {
                    prob_map[value]
                } else {
                    let parents = &node.parents;
                    let evident_parents = parents
                        .iter()
                        .enumerate()
                        .filter(|(_, parent)| evidence.contains_key(parent))
                        .map(|(i, _)| i)
                        .collect::<Vec<usize>>();
                    let mut pi = 0.0;
                    for (parent_values, prob) in &node.probability {
                        let mut skip = false;
                        for e in &evident_parents {
                            if parent_values[*e] != evidence[&parents[*e]] {
                                skip = true;
                                break;
                            }
                        }
                        if skip {
                            continue;
                        }
                        let mut parent_mul = 1.0;
                        for (i, parent) in node.parents.iter().enumerate() {
                            parent_mul *= pi_map[&(*parent, node.id)][&parent_values[i]];
                        }
                        pi += prob[value] * parent_mul;
                    }
                    pi
                };
                probs.push(lambda * pi);
            }
            let sum: Probability = probs.iter().sum();
            for (i, value) in self.value_space.iter().enumerate() {
                map.insert(value.clone(), probs[i] / sum);
            }
            inferred_probabilities.push(map);
        }
        inferred_probabilities
    }

    /// Get the inferred probability of a node.
    /// Must be called after `infer` and give return value of `infer` as argument.
    pub fn get_inferred_probability(
        &self,
        inferred_probabilities: &[HashMap<T, Probability>],
        name: &str,
        value: T,
    ) -> Probability {
        if !self.node_map.contains_key(name) {
            panic!("Node name not found");
        }
        if !self.value_space.contains(&value) {
            panic!("Value not found in value space");
        }
        inferred_probabilities[self.get_node_index(name)][&value]
    }
}
