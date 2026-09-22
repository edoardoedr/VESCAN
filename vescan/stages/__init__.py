"""One module per pipeline stage. Each module exposes:

  - run(...)             a plain function doing the actual work, importable
                          and callable from any other Python code (main.py,
                          a notebook, the Slicer Python console, ...)
  - build_arg_parser()/main()   a thin argparse CLI wrapper around run(), so
                          the stage can still be run standalone from the
                          command line for debugging one stage in isolation

No stage module should put real logic inside `if __name__ == "__main__":` -
that block only calls main().
"""