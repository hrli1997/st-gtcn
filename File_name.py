filename = "zara02"
# filename = "htl", "eth", "students03", "zara01", "zara02"

if filename == "eth":
    file_path = "datasets/ETH/seq_eth/obsmat.txt"
    val_file_path = "datasets/ETH/seq_eth/obsmat_val.txt"
    h_path = "datasets/ETH/seq_eth/H.txt"
    img_path = './datasets/ETH/seq_eth/map.png'
    reference_path = './datasets/ETH/seq_eth/reference.png'
elif filename == "htl":
    file_path = "datasets/ETH/seq_hotel/obsmat.txt"
    h_path = "datasets/ETH/seq_hotel/H.txt"
    img_path = './datasets/ETH/seq_hotel/map.png'
    reference_path = './datasets/ETH/seq_hotel/reference.png'
elif filename == "students01":
    file_path = "datasets/UCY/students01/students001.txt"
    h_path = "datasets/UCY/students01/H.txt"
    img_path = './datasets/UCY/students01/map.png'
    reference_path = './datasets/UCY/students01/reference.png'
elif filename == "students03":
    file_path = "datasets/UCY/students03/obsmat_px.txt"
    h_path = "datasets/UCY/students03/H.txt"
    img_path = './datasets/UCY/students03/map.png'
    reference_path = './datasets/UCY/students03/reference.png'
elif filename == "zara01":
    file_path = "datasets/UCY/zara01/obsmat.txt"
    h_path = "datasets/UCY/zara01/H.txt"
    img_path = './datasets/UCY/zara01/map.png'
    reference_path = './datasets/UCY/zara01/reference.png'
elif filename == "zara02":
    file_path = "datasets/UCY/zara02/obsmat.txt"
    h_path = "datasets/UCY/zara02/H.txt"
    img_path = './datasets/UCY/zara02/map.png'
    reference_path = './datasets/UCY/zara02/reference.png'
elif filename == "zara03":
    file_path = "datasets/UCY/zara03/crowds_zara03.txt"
    h_path = "datasets/UCY/zara03/H.txt"
    img_path = './datasets/UCY/zara03/map.png'
    reference_path = './datasets/UCY/zara03/reference.png'
else:
    raise Exception("didn't record this dataset")