"""

The program implements Eqs. (1)-(13) of:
    Raymond O, Chen B. ORCO: An Entropy Based Adaptive Residue-Type
    Classifier for Partial Protein NMR Chemical Shift Observations.

This is the minimal Python reference implementation. Only the standard
``math`` module is required. Inputs, reference statistics, and posteriors are
represented as Python lists so the complete calculation can be followed in
this one file.

Following the manuscript, q denotes a residue state, s a secondary-structure
state, (q,s) a fine-grained class, and r one of the 20 canonical residue types.
"""

import math


# ---------------------------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------------------------
# Each input row represents one spin system and is:
# [name, HA, CA, CB, CO, N, HN]
#
# Enter None or 0.00 for an unobserved atomic site; that site is omitted from
# the set A in Eq. (1).
# Add as many rows as needed.
SPIN_SYSTEMS = [
    ["GB1_Met1", 4.23, 54.45, 33.27, 170.54, None, 0.00],
    ["GB1_Gln2", 5.25, 55.08, 30.76, 175.92, 123.22, 8.90],
    # Deliberate end-to-end checks: no observations and one exact RefDB mean.
    ["no_information", None, None, None, None, None, None],
    ["exactly_mean_alanine_coil", 4.26, 52.84, 19.06, 177.67, 123.61, 8.15],
]

# Set these to True to print the complete residue-type and fine-class tables.
SHOW_63_CLASS_POSTERIOR = True
SHOW_20_CLASS_POSTERIOR = True


# ---------------------------------------------------------------------------
# MODEL LABELS
# ---------------------------------------------------------------------------
# Atomic-site labels in the set A of Eq. (1): HA = Halpha, CA = Calpha,
# CB = Cbeta, CO = carbonyl carbon, N = amide N, HN = amide H.
CHANNELS = ["HA", "CA", "CB", "CO", "N", "HN"]

# The 20 canonical residue types, R20 in Eqs. (9)-(11).
AMINO_ACIDS = [
    "Ala", "Arg", "Asn", "Asp", "Cys", "Gln", "Glu", "Gly", "His", "Ile",
    "Leu", "Lys", "Met", "Phe", "Pro", "Ser", "Thr", "Trp", "Tyr", "Val",
]

# The secondary-structure states s in {H,C,E}. The stored label "beta"
# corresponds to the sheet state E used in the manuscript.
SECONDARY_STRUCTURES = ["coil", "helix", "beta"]

# Numerical tolerance used only when evaluating the strict boundary in
# Eq. (13b); 1e-12 is also used by the HTML implementation.
GOLDILOCKS_EPSILON = 1e-12

# ---------------------------------------------------------------------------
# EMBEDDED REFERENCE STATISTICS
# ---------------------------------------------------------------------------
# Each row is:
# [residue state q, secondary-structure state s,
#  HA mean, HA sigma, CA mean, CA sigma, CB mean, CB sigma,
#  CO mean, CO sigma, N mean, N sigma, HN mean, HN sigma]
#
# The means and standard deviations mu_q,s,a and sigma_q,s,a are the RefDB
# values reported by Zhang, Neal, and Wishart, J. Biomol. NMR 25, 173-195
# (2003), Tables 5-10. As stated in the Methods, the only substitution is the
# aggregate BMRB distribution for Pro amide N (mu = 135.627 ppm,
# sigma = 5.150 ppm, n = 2819) in all three secondary-structure states.

REFERENCE_STATS = [
    ["Ala", "coil", 4.26, 0.33, 52.84, 1.64, 19.06, 1.26, 177.67, 1.57, 123.61, 3.77, 8.15, 0.72],
    ["Ala", "helix", 4.03, 0.33, 54.83, 1.05, 18.26, 0.88, 179.40, 1.32, 121.44, 2.37, 8.08, 0.52],
    ["Ala", "beta", 4.77, 0.55, 51.53, 1.48, 21.14, 2.05, 176.09, 1.51, 124.47, 4.39, 8.44, 0.76],
    ["Cys(ox)", "coil", 4.65, 0.39, 55.60, 2.58, 40.97, 3.19, 174.93, 1.89, 117.96, 3.88, 8.25, 0.71],
    ["Cys(ox)", "helix", 4.15, 0.67, 58.00, 2.54, 39.44, 2.86, 176.16, 1.64, 117.68, 3.33, 8.20, 0.69],
    ["Cys(ox)", "beta", 5.15, 0.51, 54.99, 2.00, 43.90, 4.18, 173.57, 1.64, 121.04, 4.53, 8.80, 0.64],
    ["Cys(red)", "coil", 4.65, 0.39, 57.53, 3.05, 29.35, 2.52, 174.93, 1.89, 117.96, 3.88, 8.25, 0.71],
    ["Cys(red)", "helix", 4.15, 0.67, 61.31, 3.50, 27.75, 2.07, 176.16, 1.64, 117.68, 3.33, 8.20, 0.69],
    ["Cys(red)", "beta", 5.15, 0.51, 56.88, 2.02, 30.16, 1.97, 173.57, 1.64, 121.04, 4.53, 8.80, 0.64],
    ["Asp", "coil", 4.60, 0.28, 54.18, 1.60, 40.85, 1.32, 176.31, 1.34, 119.95, 4.41, 8.36, 0.62],
    ["Asp", "helix", 4.43, 0.22, 56.70, 1.61, 40.51, 1.33, 178.02, 1.33, 119.22, 2.69, 8.18, 0.56],
    ["Asp", "beta", 4.94, 0.40, 53.87, 1.64, 42.30, 1.62, 175.54, 1.57, 122.17, 4.40, 8.51, 0.61],
    ["Glu", "coil", 4.28, 0.33, 56.87, 1.82, 30.20, 1.55, 176.43, 1.36, 120.43, 4.05, 8.37, 0.68],
    ["Glu", "helix", 4.01, 0.24, 59.11, 1.16, 29.37, 0.99, 178.61, 1.21, 119.04, 2.82, 8.22, 0.62],
    ["Glu", "beta", 4.78, 0.49, 55.52, 1.67, 32.01, 1.98, 175.35, 1.40, 122.09, 3.95, 8.53, 0.62],
    ["Phe", "coil", 4.54, 0.47, 57.98, 2.02, 39.45, 1.98, 175.59, 1.60, 119.67, 4.65, 8.17, 0.83],
    ["Phe", "helix", 4.16, 0.46, 60.81, 1.90, 38.78, 1.31, 177.13, 1.38, 119.16, 3.33, 8.18, 0.62],
    ["Phe", "beta", 5.09, 0.46, 56.65, 1.59, 41.54, 1.74, 174.25, 1.63, 121.08, 4.45, 8.75, 0.72],
    ["Gly", "coil", 3.96, 0.35, 45.51, 1.05, None, None, 173.89, 1.42, 109.13, 3.91, 8.33, 0.78],
    ["Gly", "helix", 3.81, 0.38, 46.91, 1.10, None, None, 175.51, 1.23, 107.51, 2.69, 8.29, 0.67],
    ["Gly", "beta", 4.20, 0.60, 45.22, 1.17, None, None, 172.55, 1.58, 109.32, 3.94, 8.34, 0.86],
    ["His", "coil", 4.53, 0.50, 55.86, 1.96, 29.97, 2.42, 174.83, 1.72, 118.72, 4.59, 8.21, 0.79],
    ["His", "helix", 4.33, 0.34, 59.04, 1.74, 29.54, 1.46, 176.98, 1.29, 117.95, 2.63, 8.10, 0.56],
    ["His", "beta", 5.06, 0.48, 55.09, 1.78, 31.85, 2.22, 174.17, 1.54, 120.49, 4.51, 8.62, 0.74],
    ["Ile", "coil", 4.15, 0.38, 61.03, 1.90, 38.65, 1.69, 175.57, 1.67, 120.87, 5.31, 7.98, 0.84],
    ["Ile", "helix", 3.67, 0.33, 64.57, 1.74, 37.60, 1.15, 177.72, 1.29, 119.71, 2.88, 8.02, 0.52],
    ["Ile", "beta", 4.68, 0.48, 60.05, 1.57, 39.86, 1.98, 174.86, 1.39, 122.85, 4.63, 8.68, 0.70],
    ["Lys", "coil", 4.26, 0.41, 56.59, 1.78, 32.79, 1.67, 176.34, 1.43, 120.45, 4.08, 8.23, 0.72],
    ["Lys", "helix", 3.99, 0.30, 58.93, 1.44, 32.27, 0.88, 178.40, 1.46, 119.20, 2.64, 7.99, 0.56],
    ["Lys", "beta", 4.69, 0.51, 55.40, 1.34, 34.63, 1.78, 175.31, 1.29, 122.21, 4.32, 8.48, 0.68],
    ["Leu", "coil", 4.36, 0.37, 54.92, 1.70, 42.38, 1.64, 176.89, 1.71, 121.48, 4.27, 8.08, 0.76],
    ["Leu", "helix", 4.00, 0.34, 57.52, 1.23, 41.65, 1.05, 178.53, 1.30, 119.61, 2.73, 8.05, 0.54],
    ["Leu", "beta", 4.82, 0.46, 54.08, 1.31, 43.79, 2.00, 175.67, 1.47, 124.05, 4.26, 8.60, 0.71],
    ["Met", "coil", 4.38, 0.41, 55.67, 1.54, 33.36, 2.26, 175.35, 1.89, 119.66, 3.95, 8.18, 0.65],
    ["Met", "helix", 4.07, 0.34, 58.09, 1.81, 32.27, 1.66, 177.95, 1.12, 118.18, 2.75, 8.09, 0.58],
    ["Met", "beta", 4.96, 0.47, 54.58, 1.24, 35.05, 2.29, 174.83, 1.40, 121.66, 4.02, 8.64, 0.67],
    ["Asn", "coil", 4.66, 0.36, 53.23, 1.51, 38.55, 1.41, 175.08, 1.51, 118.22, 4.71, 8.40, 0.78],
    ["Asn", "helix", 4.48, 0.22, 55.45, 1.42, 38.61, 1.31, 176.91, 1.55, 117.30, 2.85, 8.22, 0.58],
    ["Asn", "beta", 5.06, 0.49, 52.74, 1.47, 40.12, 2.07, 174.64, 1.65, 121.58, 4.35, 8.60, 0.64],
    ["Pro", "coil", 4.37, 0.35, 63.47, 1.26, 31.94, 0.95, 176.89, 1.34, 135.627, 5.150, None, None],
    ["Pro", "helix", 4.22, 0.29, 65.49, 1.08, 31.46, 0.95, 178.34, 1.45, 135.627, 5.150, None, None],
    ["Pro", "beta", 4.60, 0.50, 62.64, 1.03, 32.27, 1.20, 176.18, 1.40, 135.627, 5.150, None, None],
    ["Gln", "coil", 4.26, 0.34, 56.12, 1.72, 29.14, 1.69, 175.90, 1.52, 119.49, 4.11, 8.23, 0.65],
    ["Gln", "helix", 3.99, 0.28, 58.47, 1.19, 28.51, 0.92, 177.97, 1.29, 118.45, 2.84, 8.04, 0.55],
    ["Gln", "beta", 4.80, 0.49, 54.83, 1.41, 31.28, 1.93, 174.88, 1.38, 121.08, 4.13, 8.48, 0.66],
    ["Arg", "coil", 4.24, 0.43, 56.42, 1.94, 30.66, 1.67, 176.02, 1.69, 120.42, 4.33, 8.25, 0.67],
    ["Arg", "helix", 3.99, 0.32, 58.93, 1.55, 30.14, 1.14, 178.26, 1.43, 118.90, 2.83, 8.07, 0.55],
    ["Arg", "beta", 4.74, 0.50, 55.14, 1.64, 32.19, 1.80, 175.14, 1.36, 122.31, 4.25, 8.56, 0.64],
    ["Ser", "coil", 4.47, 0.35, 58.38, 1.69, 64.03, 1.27, 174.49, 1.31, 115.55, 3.95, 8.23, 0.65],
    ["Ser", "helix", 4.25, 0.25, 60.88, 1.61, 63.08, 1.12, 175.94, 1.39, 114.87, 2.99, 8.14, 0.56],
    ["Ser", "beta", 4.91, 0.48, 57.54, 1.40, 65.16, 1.51, 173.55, 1.50, 116.89, 4.02, 8.50, 0.67],
    ["Thr", "coil", 4.45, 0.36, 61.64, 2.07, 70.12, 1.33, 174.70, 1.47, 113.36, 4.98, 8.16, 0.69],
    ["Thr", "helix", 4.00, 0.34, 65.61, 2.39, 68.88, 1.17, 175.92, 1.15, 114.60, 3.99, 8.04, 0.51],
    ["Thr", "beta", 4.86, 0.46, 61.06, 1.59, 70.75, 1.51, 173.66, 1.50, 116.46, 5.00, 8.51, 0.61],
    ["Val", "coil", 4.12, 0.41, 62.06, 2.16, 32.71, 1.37, 175.66, 1.47, 119.77, 5.45, 8.04, 0.65],
    ["Val", "helix", 3.58, 0.36, 66.16, 1.55, 31.49, 0.72, 177.65, 1.38, 119.19, 3.59, 8.02, 0.65],
    ["Val", "beta", 4.60, 0.48, 60.83, 1.64, 33.91, 1.61, 174.80, 1.39, 121.90, 5.05, 8.62, 0.69],
    ["Trp", "coil", 4.55, 0.48, 57.78, 1.71, 29.67, 1.74, 176.15, 1.14, 120.16, 5.30, 7.92, 0.89],
    ["Trp", "helix", 4.38, 0.37, 60.01, 1.77, 29.30, 1.40, 178.05, 1.57, 119.84, 3.11, 8.12, 0.74],
    ["Trp", "beta", 5.19, 0.50, 56.41, 1.87, 31.50, 1.70, 175.41, 1.66, 122.09, 5.15, 8.59, 0.83],
    ["Tyr", "coil", 4.52, 0.44, 57.97, 2.17, 38.95, 1.84, 175.39, 1.67, 119.52, 5.10, 8.06, 0.77],
    ["Tyr", "helix", 4.09, 0.39, 60.98, 1.76, 38.25, 1.11, 177.36, 1.40, 119.17, 2.91, 8.07, 0.62],
    ["Tyr", "beta", 5.10, 0.54, 56.83, 1.71, 40.97, 1.85, 174.54, 1.45, 121.43, 4.78, 8.68, 0.76],
]

if len(REFERENCE_STATS) != 63:
    raise ValueError("REFERENCE_STATS must contain 63 fine-grained classes.")


def canonical_residue(residue_state):
    """Map either cysteine residue state q to canonical residue type Cys."""
    if residue_state in ("Cys(ox)", "Cys(red)"):
        return "Cys"
    return residue_state


def is_finite_number(value):
    """Return True for a finite int or float, but not a Boolean."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def validate_spin_system(spin_system):
    """Validate one [name, HA, CA, CB, CO, N, HN] input row."""
    expected_length = 1 + len(CHANNELS)
    if not isinstance(spin_system, (list, tuple)) or len(spin_system) != expected_length:
        raise ValueError(
            "Each input row must be [name, HA, CA, CB, CO, N, HN]"
        )

    for channel, value in zip(CHANNELS, spin_system[1:]):
        if value is None:
            continue
        if not is_finite_number(value):
            raise ValueError(
                "%s shift for %s must be a finite number, None, or 0.00"
                % (channel, spin_system[0])
            )

def classify(spin_system):
    """Evaluate Eqs. (2)-(13) for one spin system."""
    validate_spin_system(spin_system)

    input_shifts = spin_system[1:]

    # Each element is [residue state q, secondary-structure state s,
    # log(L_q,s)]. The unnormalized fine-grained class weights L_q,s from
    # Eq. (7) are accumulated in log space to avoid numerical underflow.
    fine_class_log_weights = []

    for reference_row in REFERENCE_STATS:
        residue_state = reference_row[0]
        secondary_state = reference_row[1]

        # Eq. (6): scoring-level residue weight w_q. Each cysteine oxidation
        # state receives half weight so canonical Cys has unit total weight.
        residue_weight = 1.0
        if residue_state in ("Cys(ox)", "Cys(red)"):
            residue_weight = 0.5

        log_class_weight = math.log(residue_weight)
        class_is_compatible = True

        for atomic_site_index in range(len(CHANNELS)):
            observed_shift = input_shifts[atomic_site_index]
            if observed_shift is None or observed_shift == 0.00:
                continue

            mean = reference_row[2 + 2 * atomic_site_index]
            sigma = reference_row[3 + 2 * atomic_site_index]

            # Below Eq. (7b): exclude a fine-grained class if an observed
            # atomic site has no reference distribution for that class.
            if mean is None:
                class_is_compatible = False
                break

            # Eq. (2): fine-grained Z-score Z_q,s,a.
            z_score = (observed_shift - mean) / sigma

            # Eq. (3): add ln p(delta_a | q,s). Summing these terms over the
            # observed atomic sites is equivalent to Eqs. (5a)-(5c).
            log_class_weight -= 0.5 * math.log(2.0 * math.pi)
            log_class_weight -= math.log(sigma)
            log_class_weight -= 0.5 * z_score * z_score

        if not class_is_compatible:
            log_class_weight = -math.inf

        # Eq. (7): log(L_q,s), including the scoring-level residue weight w_q.
        fine_class_log_weights.append(
            [residue_state, secondary_state, log_class_weight]
        )

    largest_log_weight = max(row[2] for row in fine_class_log_weights)
    if not math.isfinite(largest_log_weight):
        raise ValueError(
            "No fine-grained class is compatible with the observed atomic sites"
        )

    # Subtracting the maximum log weight prevents underflow. This common
    # scaling factor cancels during the normalization in Eq. (9).
    fine_class_weights = []
    for residue_state, secondary_state, log_class_weight in fine_class_log_weights:
        scaled_weight = 0.0
        if math.isfinite(log_class_weight):
            scaled_weight = math.exp(log_class_weight - largest_log_weight)
        fine_class_weights.append(
            [residue_state, secondary_state, scaled_weight]
        )

    normalizing_sum = math.fsum(row[2] for row in fine_class_weights)
    if not math.isfinite(normalizing_sum) or normalizing_sum <= 0.0:
        raise ValueError("ORCO posterior normalization failed")

    residue_scores = []
    for amino_acid in AMINO_ACIDS:
        residue_scores.append([amino_acid, 0.0])

    secondary_state_scores = []
    for secondary_state in SECONDARY_STRUCTURES:
        secondary_state_scores.append([secondary_state, 0.0])

    for residue_state, secondary_state, scaled_weight in fine_class_weights:
        residue_type = canonical_residue(residue_state)

        # Eqs. (8a)-(8b): marginalize over secondary structure and merge the
        # reduced and oxidized cysteine states into canonical Cys.
        residue_index = AMINO_ACIDS.index(residue_type)
        residue_scores[residue_index][1] += scaled_weight

        # Auxiliary marginal over the latent secondary-structure state s. It
        # is returned for inspection but is not evaluated as a prediction in
        # the manuscript.
        secondary_state_index = SECONDARY_STRUCTURES.index(secondary_state)
        secondary_state_scores[secondary_state_index][1] += scaled_weight

    # Eq. (9): normalize the residue-level scores L_r over R20.
    residue_posterior = []
    for residue_type, score in residue_scores:
        residue_posterior.append([residue_type, score / normalizing_sum])

    secondary_state_posterior = []
    for secondary_state, score in secondary_state_scores:
        secondary_state_posterior.append(
            [secondary_state, score / normalizing_sum]
        )

    # Auxiliary normalized decomposition over the 63 fine-grained classes.
    fine_class_posterior = []
    for residue_state, secondary_state, scaled_weight in fine_class_weights:
        fine_class_posterior.append(
            [residue_state, secondary_state, scaled_weight / normalizing_sum]
        )

    # Eq. (10): rank residue types by decreasing posterior probability.
    # Python's stable sort preserves AMINO_ACIDS order for exact ties.
    ranked_residue_posterior = sorted(
        residue_posterior,
        key=lambda row: row[1],
        reverse=True,
    )
    ranked_fine_classes = sorted(
        fine_class_posterior,
        key=lambda row: row[2],
        reverse=True,
    )

    # Eq. (11): Shannon entropy H(delta_A) of the residue-type posterior.
    # Natural logarithms give entropy in nats.
    entropy_terms = []
    for residue_type, probability in residue_posterior:
        if probability > 0.0:
            entropy_terms.append(probability * math.log(probability))
    entropy = -math.fsum(entropy_terms)

    # Eq. (12): entropy-derived cumulative-probability cutoff c(delta_A).
    cutoff = 1.0 - math.exp(-entropy)

    # Eq. (13): the Goldilocks set G(delta_A) is the smallest ranked prefix
    # whose cumulative posterior probability strictly exceeds the cutoff.
    goldilocks_set = []
    cumulative_probability = 0.0
    for residue_type, probability in ranked_residue_posterior:
        goldilocks_set.append([residue_type, probability])
        cumulative_probability += probability
        if cumulative_probability > cutoff + GOLDILOCKS_EPSILON:
            break

    return (ranked_residue_posterior, secondary_state_posterior,
            ranked_fine_classes, goldilocks_set, entropy, cutoff)


def print_result(spin_system):
    """Print one ORCO report."""
    name = spin_system[0]
    input_shifts = spin_system[1:]

    (ranked_residue_posterior, secondary_state_posterior,
     ranked_fine_classes, goldilocks_set, entropy, cutoff) = classify(spin_system)

    # Calculations use full precision; formatting rounds only the display.
    print("=" * 72)
    print("ORCO result:", name)
    print("Observed shifts:")
    for channel, value in zip(CHANNELS, input_shifts):
        if value is not None and value != 0.00:
            print("  %2s = %.4f ppm" % (channel, value))

    if SHOW_20_CLASS_POSTERIOR:
        print("\nCanonical 20-residue posterior (Eqs. 8-9):")
        for residue_type, probability in ranked_residue_posterior:
            print("  %s  %4g" % (residue_type, probability))

    if SHOW_63_CLASS_POSTERIOR:
        print("\nFull 63-class posterior (q,s):")
        for residue_state, secondary_state, probability in ranked_fine_classes:
            print("  %-8s %-5s  %.4g" % (residue_state, secondary_state, probability))

    print("\nGoldilocks set (Eq. 13):")
    print("  " + ", ".join(
        "%s (%.4g)" % (row[0], row[1]) for row in goldilocks_set
    ))


if __name__ == "__main__":
    for spin_system in SPIN_SYSTEMS:
        print_result(spin_system)