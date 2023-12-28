mod bayesian_network;
use bayesian_network::*;
use std::collections::HashMap;

fn main() {
    let mut network = BayesianNetwork::new(vec![true, false]);
    let mut a_prob_map = HashMap::new();
    a_prob_map.insert(true, 0.01);
    a_prob_map.insert(false, 0.99);
    network.add_node("業績", NodeType::Root(a_prob_map));
    let mut b_prob_map = HashMap::new();
    b_prob_map.insert(true, 0.1);
    b_prob_map.insert(false, 0.9);
    network.add_node("競馬", NodeType::Root(b_prob_map));
    network.add_node("ごきげん", NodeType::Intermediate);
    let mut c_prob_map = HashMap::new();
    let mut c_prob_map_tt = HashMap::new();
    c_prob_map_tt.insert(true, 0.99);
    c_prob_map_tt.insert(false, 0.01);
    let mut c_prob_map_ft = HashMap::new();
    c_prob_map_ft.insert(true, 0.6);
    c_prob_map_ft.insert(false, 0.4);
    let mut c_prob_map_tf = HashMap::new();
    c_prob_map_tf.insert(true, 0.9);
    c_prob_map_tf.insert(false, 0.1);
    let mut c_prob_map_ff = HashMap::new();
    c_prob_map_ff.insert(true, 0.01);
    c_prob_map_ff.insert(false, 0.99);
    c_prob_map.insert(vec![true, true], c_prob_map_tt);
    c_prob_map.insert(vec![false, true], c_prob_map_ft);
    c_prob_map.insert(vec![true, false], c_prob_map_tf);
    c_prob_map.insert(vec![false, false], c_prob_map_ff);
    network.add_dependency(vec!["業績", "競馬"], "ごきげん", c_prob_map);
    network.add_node("ボーナス", NodeType::Leaf);
    network.add_node("ごちそう", NodeType::Leaf);
    let mut d_prob_map = HashMap::new();
    let mut d_prob_map_t = HashMap::new();
    d_prob_map_t.insert(true, 0.3);
    d_prob_map_t.insert(false, 0.7);
    let mut d_prob_map_f = HashMap::new();
    d_prob_map_f.insert(true, 0.01);
    d_prob_map_f.insert(false, 0.99);
    d_prob_map.insert(vec![true], d_prob_map_t);
    d_prob_map.insert(vec![false], d_prob_map_f);
    network.add_dependency(vec!["ごきげん"], "ボーナス", d_prob_map);
    let mut e_prob_map = HashMap::new();
    let mut e_prob_map_t = HashMap::new();
    e_prob_map_t.insert(true, 0.9);
    e_prob_map_t.insert(false, 0.1);
    let mut e_prob_map_f = HashMap::new();
    e_prob_map_f.insert(true, 0.01);
    e_prob_map_f.insert(false, 0.99);
    e_prob_map.insert(vec![true], e_prob_map_t);
    e_prob_map.insert(vec![false], e_prob_map_f);
    network.add_dependency(vec!["ごきげん"], "ごちそう", e_prob_map);

    let mut evidence = HashMap::new();
    evidence.insert("ボーナス", true);

    let inferred_probabilities = network.infer(&evidence);
    println!(
        "業績が上がった確率: {}",
        network.get_inferred_probability(&inferred_probabilities, "業績", true)
    );
    println!(
        "競馬で勝った確率: {}",
        network.get_inferred_probability(&inferred_probabilities, "競馬", true)
    );
    println!(
        "ごきげんな確率: {}",
        network.get_inferred_probability(&inferred_probabilities, "ごきげん", true)
    );
    println!(
        "ボーナスが出る確率: {}",
        network.get_inferred_probability(&inferred_probabilities, "ボーナス", true)
    );
    println!(
        "ごちそうが出る確率: {}",
        network.get_inferred_probability(&inferred_probabilities, "ごちそう", true)
    );

    println!("{:?}", network);
}
