# neural-wasm

Bibliothèque WebAssembly minimale.

## Build

```bash
./build.sh
```

## Test

Ouvrir `../www/index.html` dans un navigateur après le build.


#

download from
<https://www.openml.org/search?type=data&sort=runs&id=554&status=active>

I applied 
```bash
grep -v "^@" mnist.arff | grep -v "^%" | grep -v "^$" > mnist.csv
```
