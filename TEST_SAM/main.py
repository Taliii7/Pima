from Sam import *



if __name__ == "__main__":

# le permier lancement est un peu long, à cause de la methode modelAvailable qui cherche le type de gpu, mais après c'est plus rapide
    print("\n---------------TEST SAM----------------\n")
    masks = run_sam_inference("images","results")
    #masksSegmentAll= segmentAll("images/image")
    print ("------------FIN TEST SAM----------------")
