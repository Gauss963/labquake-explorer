"""Pure numerical analysis for Labquake Explorer.

Functions here take arrays in and return dataclasses or dicts; they know
nothing about Tk, the data tree, or persistence.  Views call them and decide
what to display and save.
"""
