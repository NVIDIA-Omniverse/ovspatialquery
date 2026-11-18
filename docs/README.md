# ovspatialquery Documentation

The ovspatialquery user guide explains how to install the library, publish geometry, and query
it from Python and C. Source releases include the built guide under `docs/html/`. Open
`docs/html/index.html` in a browser, or serve it as described in
[Read the Guide Locally](#read-the-guide-locally).

## Find a Topic

The following table lists where each topic starts, relative to `docs/html/`:

| Topic | Page |
| --- | --- |
| Getting started: install the wheel or the SDK and check the installation | `install.html` |
| Your first query in Python and in C | `first-query.html` |
| Concepts: populations, versions, views, results, batches, and pairs | `concepts/index.html` |
| Examples: the rendered lessons in Python and C++ | `examples.html` |
| C API reference, generated from the public headers | `reference/c-core.html` |
| Python API reference | `reference/python.html` |
| Using the agent skill with a coding agent | `agents.html` |
| Troubleshooting: the messages a first installation or program most often meets | `troubleshooting.html` |

The public C headers under [`include/ovspatialquery/`](../include/ovspatialquery/) are the
contract that the C API reference is generated from.

## Read the Guide Locally

To serve the built guide, run this command from the repository root:

```bash
./dev.sh docs --serve
```

Then open `http://127.0.0.1:8000/`. On Windows, run `dev.cmd docs --serve` instead.
