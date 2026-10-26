"""Small pure-Python variational graph autoencoder, for investigation only.

This is an experiment inspired by hypergraph/VAE knowledge-recombination work,
not a reproduction of H2GLM or a mathematical inference engine. Maximum 124
nodes; no text embedding, LLM, remote service, dependency install, or GPU claim.
"""
from __future__ import annotations

import math
import random

from discovery import bounded_integer
from frontier_ir import digest, validate_ir

MAX_NODES = 124
FEATURE_NAMES = ['bias', 'incoming_degree', 'outgoing_degree', 'neighbor_degree',
                 'copremise_degree', 'mean_training_hyperedge_arity',
                 'incident_training_hyperedges', 'training_root_indicator']


def training_features(node_ids, training_edges):
    """Rebuild higher-order incidence features from surviving training edges.

    An edge is [premise, conclusion]. Each conclusion groups its surviving
    premises into a training hyperedge; original arity, held-out incidence,
    titles, topics, scopes and acceptance states are never feature inputs.
    """
    if type(node_ids) is not list or not 1 <= len(node_ids) <= MAX_NODES:
        raise ValueError('feature graph must contain 1 to 124 nodes')
    if any(type(n) is not str or not n or len(n) > 256 for n in node_ids):
        raise ValueError('invalid feature node ID')
    if len(set(node_ids)) != len(node_ids):
        raise ValueError('duplicate feature node')
    if type(training_edges) is not list or len(training_edges) > 2048:
        raise ValueError('training edges bound exceeded')
    ids = set(node_ids)
    pairs = set()
    for pair in training_edges:
        if type(pair) not in (list, tuple) or len(pair) != 2 or any(type(n) is not str or n not in ids for n in pair) or pair[0] == pair[1]:
            raise ValueError('invalid training edge')
        if tuple(pair) in pairs:
            raise ValueError('duplicate training edge')
        pairs.add(tuple(pair))
    incoming = {n: set() for n in ids}
    outgoing = {n: set() for n in ids}
    for source, target in pairs:
        incoming[target].add(source)
        outgoing[source].add(target)
    copremises = {n: set() for n in ids}
    arities = {n: [] for n in ids}
    for target in sorted(ids):
        parents = incoming[target]
        if not parents:
            continue
        members = parents | {target}
        for member in members:
            arities[member].append(len(members))
        for member in parents:
            copremises[member].update(parents - {member})
    scale = max(1, len(ids) - 1)
    features = []
    for node in node_ids:
        groups = arities[node]
        features.append([1.0, len(incoming[node]) / scale, len(outgoing[node]) / scale,
            len(incoming[node] | outgoing[node]) / scale, len(copremises[node]) / scale,
            (sum(groups) / len(groups) / len(ids)) if groups else 0.0,
            len(groups) / scale, 0.0 if incoming[node] else 1.0])
    return features


def _sigmoid(value):
    return 1 / (1 + math.exp(-max(-60.0, min(60.0, value))))


def _dot(first, second):
    return sum(a * b for a, b in zip(first, second))


def _auc(positive, negative):
    if not positive or not negative:
        return None
    return sum(1.0 if p > n else 0.5 if p == n else 0.0
               for p in positive for n in negative) / (len(positive) * len(negative))


def train_experiment(ir, seed=0, epochs=100):
    """Train a diagonal-Gaussian linear encoder with a dot-product decoder.

    Objective: balanced training-edge Bernoulli reconstruction plus 0.01 times
    mean KL(q(z|X)||N(0,I)). The reparameterization trick trains both mean and
    log-variance weights with deterministic seeded Gaussian samples. Evaluation
    uses latent means. Directed labels are evaluated, although the symmetric
    decoder cannot distinguish direction; this is an explicit model limitation.
    """
    validate_ir(ir)
    bounded_integer(seed, 'seed', -(2**31), 2**31 - 1)
    bounded_integer(epochs, 'epochs', 1, 500)
    nodes = sorted(n['id'] for n in ir['nodes'])
    if not 2 <= len(nodes) <= MAX_NODES:
        raise ValueError('latent experiment limited to 2 through 124 nodes')
    index = {node: i for i, node in enumerate(nodes)}
    all_edges = sorted((p, edge['conclusion']) for edge in ir['hyperedges'] for p in edge['premises'])
    if not 5 <= len(all_edges) <= 2048:
        raise ValueError('insufficient or excessive training edge count')
    rng = random.Random(seed)
    shuffled = all_edges[:]
    rng.shuffle(shuffled)
    heldout_count = max(1, len(shuffled) // 5)
    heldout = sorted(shuffled[:heldout_count])
    train = sorted(shuffled[heldout_count:])
    positive_set = set(all_edges)
    # The decoder is symmetric. Reserve unordered pairs, so a reversed
    # evaluation pair cannot appear as any training example.
    positive_unordered = {tuple(sorted(p)) for p in all_edges}
    absent = [(a, b) for i, a in enumerate(nodes) for b in nodes[i + 1:]
              if (a, b) not in positive_unordered]
    absent = [(b, a) if rng.random() < 0.5 else (a, b) for a, b in absent]
    if len(absent) < len(all_edges):
        raise ValueError('insufficient disjoint negative examples')
    rng.shuffle(absent)
    heldout_negative = sorted(absent[:heldout_count])
    train_negative = sorted(absent[heldout_count:heldout_count + len(train)])
    train_lists = [list(p) for p in train]
    features = training_features(nodes, train_lists)
    width, latent = len(FEATURE_NAMES), 4
    mean_weights = [[rng.gauss(0, 0.15) for _ in range(latent)] for _ in range(width)]
    variance_weights = [[rng.gauss(0, 0.03) for _ in range(latent)] for _ in range(width)]
    initial_hash = digest([mean_weights, variance_weights])
    samples = [(index[a], index[b], 1.0) for a, b in train] + [(index[a], index[b], 0.0) for a, b in train_negative]
    beta, learning_rate = 0.01, 0.08
    losses = []

    def encode():
        means = [[sum(row[f] * mean_weights[f][j] for f in range(width)) for j in range(latent)] for row in features]
        raw_logvar = [[sum(row[f] * variance_weights[f][j] for f in range(width)) for j in range(latent)] for row in features]
        logvar = [[max(-6.0, min(2.0, value)) for value in row] for row in raw_logvar]
        return means, logvar, raw_logvar

    initial_means, _, _ = encode()
    initial_train_auc = _auc(
        [_dot(initial_means[index[a]], initial_means[index[b]]) for a, b in train],
        [_dot(initial_means[index[a]], initial_means[index[b]]) for a, b in train_negative])
    for epoch in range(epochs):
        means, logvar, raw_logvar = encode()
        noise = [[rng.gauss(0, 1) for _ in range(latent)] for _ in nodes]
        deviation = [[math.exp(value / 2) for value in row] for row in logvar]
        z = [[means[i][j] + deviation[i][j] * noise[i][j] for j in range(latent)] for i in range(len(nodes))]
        gradient_z = [[0.0] * latent for _ in nodes]
        reconstruction = 0.0
        for a, b, label in samples:
            logit = _dot(z[a], z[b])
            reconstruction += max(logit, 0) - label * logit + math.log1p(math.exp(-abs(logit)))
            delta = (_sigmoid(logit) - label) / len(samples)
            for j in range(latent):
                gradient_z[a][j] += delta * z[b][j]
                gradient_z[b][j] += delta * z[a][j]
        kl = sum(0.5 * (math.exp(logvar[i][j]) + means[i][j]**2 - 1 - logvar[i][j])
                 for i in range(len(nodes)) for j in range(latent)) / len(nodes)
        loss = reconstruction / len(samples) + beta * kl
        if not math.isfinite(loss):
            raise ValueError('nonfinite latent training objective')
        if epoch in {0, epochs - 1}:
            losses.append({'epoch': epoch + 1, 'objective': loss,
                           'reconstruction': reconstruction / len(samples), 'kl': kl})
        gradient_mean = [[0.0] * latent for _ in range(width)]
        gradient_variance = [[0.0] * latent for _ in range(width)]
        for i, row in enumerate(features):
            for j in range(latent):
                dm = gradient_z[i][j] + beta * means[i][j] / len(nodes)
                dv = (gradient_z[i][j] * 0.5 * deviation[i][j] * noise[i][j]
                      + beta * 0.5 * (math.exp(logvar[i][j]) - 1) / len(nodes))
                if raw_logvar[i][j] < -6 or raw_logvar[i][j] > 2:
                    dv = 0.0
                for f in range(width):
                    gradient_mean[f][j] += row[f] * dm
                    gradient_variance[f][j] += row[f] * dv
        for f in range(width):
            for j in range(latent):
                mean_weights[f][j] -= learning_rate * max(-5.0, min(5.0, gradient_mean[f][j]))
                variance_weights[f][j] -= learning_rate * max(-5.0, min(5.0, gradient_variance[f][j]))
    means, _, _ = encode()
    indegree, outdegree = dict.fromkeys(nodes, 0), dict.fromkeys(nodes, 0)
    for a, b in train:
        outdegree[a] += 1
        indegree[b] += 1

    def score(pair):
        return _sigmoid(_dot(means[index[pair[0]]], means[index[pair[1]]]))

    def baseline(pair):
        return (1 + outdegree[pair[0]]) * (1 + indegree[pair[1]])

    latent_auc = _auc([score(p) for p in heldout], [score(p) for p in heldout_negative])
    degree_auc = _auc([baseline(p) for p in heldout], [baseline(p) for p in heldout_negative])
    train_auc = _auc([score(p) for p in train], [score(p) for p in train_negative])
    better = latent_auc > degree_auc
    # A single small split cannot qualify a model for production graph ranking.
    # Expose the measured comparison while keeping independent structural ranking.
    candidates = []
    evaluation_pairs = set(heldout) | set(heldout_negative)
    train_negative_set = set(train_negative)
    for a, b in absent:
        if (a, b) in evaluation_pairs or (a, b) in train_negative_set:
            continue
        candidates.append({'source_id': a, 'target_id': b,
            'score': score((a, b)), 'degree_score': baseline((a, b)),
            'status': 'UNVERIFIED_MODEL_SCORE', 'certified': False, 'installable': False})
    candidates.sort(key=lambda c: (-c['score'], c['source_id'], c['target_id']))
    model = {'kind': 'linear_diagonal_gaussian_variational_graph_autoencoder',
             'latent_dimensions': latent, 'feature_names': FEATURE_NAMES[:],
             'mean_weights': mean_weights, 'log_variance_weights': variance_weights,
             'parameters_sha256': digest([mean_weights, variance_weights]),
             'decoder': 'symmetric sigmoid latent inner product'}
    return {'schema_version': 'UL-frontier-learning-1', 'status': 'BOUNDED_PREDICTION_EXPERIMENT_NOT_PROOF',
            'ir_sha256': ir['ir_sha256'], 'seed': seed, 'node_ids': nodes,
            'node_count': len(nodes), 'maximum_nodes': MAX_NODES,
            'split': {'method': 'seeded 80/20 directed-edge split; equal absent-pair negatives; unordered pairs disjoint across train/evaluation',
                'train_positive_edges': train_lists,
                'train_negative_edges': [list(p) for p in train_negative],
                'heldout_positive_edges': [list(p) for p in heldout],
                'heldout_negative_edges': [list(p) for p in heldout_negative]},
            'features_sha256': digest(features), 'training_graph_sha256': digest(train_lists),
            'leakage_control': {'feature_source': 'training_positive_edges_only',
                'original_hyperedge_arity_used': False, 'node_metadata_used': False,
                'heldout_examples_used_for_gradient': False,
                'negative_sampling': 'full positive set used only to exclude false negative labels in either direction; heldout unordered pairs disjoint from training'},
            'training': {'epochs': epochs, 'learning_rate': learning_rate, 'kl_weight': beta,
                'initial_parameters_sha256': initial_hash, 'loss_samples': losses},
            'model': model, 'metrics': {'latent_heldout_auc': latent_auc,
                'degree_baseline_auc': degree_auc, 'latent_train_auc': train_auc,
                'latent_initial_train_auc': initial_train_auc,
                'chance_auc': 0.5, 'heldout_positive_count': len(heldout),
                'heldout_negative_count': len(heldout_negative), 'model_beats_degree_baseline': better},
            'qualified_for_ranking': False, 'ranking_policy': 'structural_heuristic_baseline_only',
            'qualification_reason': 'One small random split is insufficient external validation, even when its measured AUC improves.',
            'limitations': ['No mathematical validity or novelty follows from link prediction.',
                'Transductive edge holdout does not measure new-topic or new-node generalization.',
                'Symmetric decoder cannot model directed implication.',
                'No reproduction of H2GLM published architecture, data or reported efficacy.'],
            'candidate_scores': candidates[:100], 'full_theorem_certificates': 0}
