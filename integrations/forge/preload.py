"""
Preload script for ASPIRE Forge extension.
Runs before the main UI loads.
"""

def preload(parser):
    """Add command-line arguments for ASPIRE."""
    parser.add_argument(
        "--aspire-teacher",
        type=str,
        default="claude",
        help="Default ASPIRE teacher model (claude, gpt4v, local)",
    )
    parser.add_argument(
        "--aspire-critic-path",
        type=str,
        default=None,
        help="Path to pre-trained ASPIRE critic model",
    )
    parser.add_argument(
        "--aspire-cache-dir",
        type=str,
        default=None,
        help="Directory to cache ASPIRE dialogue data",
    )
