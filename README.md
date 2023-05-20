# bert-d

D bindings to [bert.cpp](https://github.com/skeskinen/bert.cpp)

## run example

1. get a bert ggml model (see [bert.cpp readme](https://github.com/skeskinen/bert.cpp/blob/master/README.md#usage))
2. run example:

    ```sh
    dub run -b release -- -m /path/to/ggml-minilm-l12-msmarco-cos-v5-q4_1.bin -p "The delicious cheese grills in the warm oven of the inviting house."
    ```
