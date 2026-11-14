def build_features(hands):
    """Convert up to two hands of 21 [x, y, z] points into 126 numbers."""
    features = [0.0] * 126
    hands = sorted(hands, key=lambda hand: hand[0][0])[:2]

    for hand_index, hand in enumerate(hands):
        wrist_x, wrist_y, wrist_z = hand[0]
        points = [
            (x - wrist_x, y - wrist_y, z - wrist_z)
            for x, y, z in hand
        ]
        scale = max(
            (max(abs(x), abs(y)) for x, y, _ in points),
            default=0.0,
        ) or 1.0

        for point_index, (x, y, z) in enumerate(points):
            offset = hand_index * 63 + point_index * 3
            features[offset:offset + 3] = [
                x / scale,
                y / scale,
                z / scale,
            ]

    return features