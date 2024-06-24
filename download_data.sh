mkdir gene_benchmarks

#open-web-math
wget -O gene_benchmarks/open-web-math.jsonl.zst https://huggingface.co/datasets/EleutherAI/proof-pile-2/blob/main/open-web-math/test/test.jsonl.zst 

#MBXP
mkdir gene_benchmarks/mbxp
wget -O "gene_benchmarks/mbxp/0000(0).jsonl" https://github.com/amazon-science/mxeval/blob/main/data/mbxp/mbcpp_release_v1.2.jsonl 
wget -O "gene_benchmarks/mbxp/0000(1).jsonl" https://github.com/amazon-science/mxeval/blob/main/data/mbxp/mbcsp_release_v1.2.jsonl 
wget -O "gene_benchmarks/mbxp/0000(2).jsonl" https://github.com/amazon-science/mxeval/blob/main/data/mbxp/mbcgp_release_v1.2.jsonl 
wget -O "gene_benchmarks/mbxp/0000(3).jsonl" https://github.com/amazon-science/mxeval/blob/main/data/mbxp/mbcplp_release_v1.2.jsonl
wget -O "gene_benchmarks/mbxp/0000(4).jsonl" https://github.com/amazon-science/mxeval/blob/main/data/mbxp/mbcscp_release_v1.2.jsonl
wget -O "gene_benchmarks/mbxp/0000(5).jsonl" https://github.com/amazon-science/mxeval/blob/main/data/mbxp/mbcswp_release_v1.2.jsonl
wget -O "gene_benchmarks/mbxp/0000(6).jsonl" https://github.com/amazon-science/mxeval/blob/main/data/mbxp/mbpp_release_v1.2.jsonl
wget -O "gene_benchmarks/mbxp/0000(7).jsonl" https://github.com/amazon-science/mxeval/blob/main/data/mbxp/mbjp_release_v1.2.jsonl
wget -O "gene_benchmarks/mbxp/0000(8).jsonl" https://github.com/amazon-science/mxeval/blob/main/data/mbxp/mbjsp_release_v1.2.jsonl
wget -O "gene_benchmarks/mbxp/0000(9).jsonl" https://github.com/amazon-science/mxeval/blob/main/data/mbxp/mbtsp_release_v1.2.jsonl
wget -O "gene_benchmarks/mbxp/0000(10).jsonl" https://github.com/amazon-science/mxeval/blob/main/data/mbxp/mbkp_release_v1.2.jsonl
wget -O "gene_benchmarks/mbxp/0000(11).jsonl" https://github.com/amazon-science/mxeval/blob/main/data/mbxp/mbrbp_release_v1.2.jsonl
wget -O "gene_benchmarks/mbxp/0000(12).jsonl" https://github.com/amazon-science/mxeval/blob/main/data/mbxp/mbphp_release_v1.2.jsonl
