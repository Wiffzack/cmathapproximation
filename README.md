# cmathapproximation
A collection of math function approximation from around the web .
>Please consider not all function work correctly and if the accuracy is as less as possible .


Usage : 
>gcc whatever.c -Dapprox -Dncp


Some Flags recommendation(for Intel CPU):

>CFLAGS="-DCLS=$(getconf LEVEL1_DCACHE_LINESIZE) -march=native -mtune=native -lm -O1 -funroll-loops -ftree-vectorize -mveclibabi=svml -msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -mavx -fprefetch-loop-arrays  -ftree-parallelize-loops=$(cat /proc/cpuinfo | awk '/^processor/{print $3}' | wc -l) -mrtm -fipa-cp -ftree-loop-distribution -fprefetch-loop-arrays -freorder-blocks-and-partition -fif-conversion -ftree-loop-if-convert -freorder-blocks -ftree-loop-if-convert -ftree-vectorize -fweb  -ftree-loop-if-convert -fsplit-wide-types -ftree-slp-vectorize -ftree-dse -fgcse-las -fsched-dep-count-heuristic -fno-tree-slsr -fsched-spec-load -fconserve-stack -fstrict-aliasing -free -ftree-vrp -fthread-jumps -maccumulate-outgoing-args -msseregparm -minline-all-stringops --param=max-reload-search-insns=356 --param=max-cselib-memory-locations=1200

The wine build contain only libwine.dll wined3d.dll and ddraw.dll and used [https://github.com/adolfintel/wined3d4win](wined3d4win)
