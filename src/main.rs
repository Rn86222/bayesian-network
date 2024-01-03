mod bayesian_network;
use bayesian_network::*;
use std::collections::HashMap;

fn example_of_morphological_analysis() {
    println!("Time flies like an arrow の品詞解析");
    let noun = "名詞";
    let verb = "動詞";
    let adjective = "形容詞";
    let article = "冠詞";
    let preposition = "前置詞";
    let mut network = BayesianNetwork::new(vec![
        noun,
        verb,
        adjective,
        article,
        preposition,
        "time",
        "flies",
        "like",
        "an",
        "arrow",
    ]);

    let mut bos_prob_map = HashMap::new();
    bos_prob_map.insert(noun, 0.6);
    bos_prob_map.insert(article, 0.4);
    network.add_node("TimePart", NodeType::Root(bos_prob_map));
    network.add_node("TimeWord", NodeType::Leaf);
    network.add_node("FliesPart", NodeType::Inner);
    network.add_node("FliesWord", NodeType::Leaf);
    network.add_node("LikePart", NodeType::Inner);
    network.add_node("LikeWord", NodeType::Leaf);
    network.add_node("AnPart", NodeType::Inner);
    network.add_node("AnWord", NodeType::Leaf);
    network.add_node("ArrowPart", NodeType::Inner);
    network.add_node("ArrowWord", NodeType::Leaf);

    let mut noun_prob_map = HashMap::new();
    noun_prob_map.insert(noun, 0.3);
    noun_prob_map.insert(verb, 0.4);
    noun_prob_map.insert(adjective, 0.1);
    noun_prob_map.insert(preposition, 0.2);

    let mut verb_prob_map = HashMap::new();
    verb_prob_map.insert(noun, 0.1);
    verb_prob_map.insert(adjective, 0.5);
    verb_prob_map.insert(article, 0.2);
    verb_prob_map.insert(preposition, 0.2);

    let mut adjective_prob_map = HashMap::new();
    adjective_prob_map.insert(noun, 0.5);
    adjective_prob_map.insert(adjective, 0.4);
    adjective_prob_map.insert(article, 0.1);

    let mut article_prob_map = HashMap::new();
    article_prob_map.insert(noun, 0.7);
    article_prob_map.insert(preposition, 0.3);

    let mut preposition_prob_map = HashMap::new();
    preposition_prob_map.insert(noun, 0.6);
    preposition_prob_map.insert(adjective, 0.1);
    preposition_prob_map.insert(article, 0.3);

    let mut a_prob_map = HashMap::new();
    a_prob_map.insert(vec![noun], noun_prob_map);
    a_prob_map.insert(vec![verb], verb_prob_map);
    a_prob_map.insert(vec![adjective], adjective_prob_map);
    a_prob_map.insert(vec![article], article_prob_map);
    a_prob_map.insert(vec![preposition], preposition_prob_map);

    network.add_dependency(vec!["TimePart"], "FliesPart", a_prob_map.clone());
    network.add_dependency(vec!["FliesPart"], "LikePart", a_prob_map.clone());
    network.add_dependency(vec!["LikePart"], "AnPart", a_prob_map.clone());
    network.add_dependency(vec!["AnPart"], "ArrowPart", a_prob_map);

    let mut noun_b_prob_map = HashMap::new();
    noun_b_prob_map.insert("time", 0.6);
    noun_b_prob_map.insert("arrow", 0.3);
    noun_b_prob_map.insert("flies", 0.1);

    let mut verb_b_prob_map = HashMap::new();
    verb_b_prob_map.insert("like", 0.7);
    verb_b_prob_map.insert("arrow", 0.1);
    verb_b_prob_map.insert("flies", 0.2);

    let mut adjective_b_prob_map = HashMap::new();
    adjective_b_prob_map.insert("like", 1.0);

    let mut article_b_prob_map = HashMap::new();
    article_b_prob_map.insert("an", 1.0);

    let mut preposition_b_prob_map = HashMap::new();
    preposition_b_prob_map.insert("like", 1.0);

    let mut b_prob_map = HashMap::new();
    b_prob_map.insert(vec![noun], noun_b_prob_map);
    b_prob_map.insert(vec![verb], verb_b_prob_map);
    b_prob_map.insert(vec![adjective], adjective_b_prob_map);
    b_prob_map.insert(vec![article], article_b_prob_map);
    b_prob_map.insert(vec![preposition], preposition_b_prob_map);

    network.add_dependency(vec!["TimePart"], "TimeWord", b_prob_map.clone());
    network.add_dependency(vec!["FliesPart"], "FliesWord", b_prob_map.clone());
    network.add_dependency(vec!["LikePart"], "LikeWord", b_prob_map.clone());
    network.add_dependency(vec!["AnPart"], "AnWord", b_prob_map.clone());
    network.add_dependency(vec!["ArrowPart"], "ArrowWord", b_prob_map);

    let mut evidence = HashMap::new();
    evidence.insert("TimeWord", "time");
    evidence.insert("FliesWord", "flies");
    evidence.insert("LikeWord", "like");
    evidence.insert("AnWord", "an");
    evidence.insert("ArrowWord", "arrow");

    let inferred_probabilities = network.infer(&evidence);
    println!(
        "Time  名詞の確率: {}",
        network.get_inferred_probability(&inferred_probabilities, "TimePart", noun)
    );
    println!(
        "flies 動詞の確率: {}  名詞の確率: {}",
        network.get_inferred_probability(&inferred_probabilities, "FliesPart", verb),
        network.get_inferred_probability(&inferred_probabilities, "FliesPart", noun)
    );
    println!(
        "like  前置詞の確率: {} 動詞の確率: {}",
        network.get_inferred_probability(&inferred_probabilities, "LikePart", preposition),
        network.get_inferred_probability(&inferred_probabilities, "LikePart", verb)
    );
    println!(
        "an    冠詞の確率: {}",
        network.get_inferred_probability(&inferred_probabilities, "AnPart", article)
    );
    println!(
        "arrow 名詞の確率: {}",
        network.get_inferred_probability(&inferred_probabilities, "ArrowPart", noun)
    );
}

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

    network.add_node("ごきげん", NodeType::Inner);

    network.add_node("ボーナス", NodeType::Leaf);

    network.add_node("ごちそう", NodeType::Leaf);

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

    example_of_morphological_analysis();
}
