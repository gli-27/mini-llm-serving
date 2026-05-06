"""Speculative decoding pipeline for accelerated inference.

Uses a small draft model to generate K candidate tokens, then the
large target model verifies all K in a single forward pass. Rejection
sampling ensures the output distribution is mathematically identical
to the target model alone.
"""
