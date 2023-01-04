# Embedding-mixer version 0.1

Similar to Embedding Inspector, more powerful but not user-friendly, text-only, for advanced users. Not tested, provided as-is.

# Formula examples

---
	emb('cat')		# retrieves the vector of internal embedding 'cat</w>'
	emb('#2368')	# same as above, but with its ID
	emb('mdjrny-ppc')	# retrieves this loaded embedding (may contain multiple vectors)

	# mix embeddings by adding their vector values (if vector counts differ, pad with zeros)
	mix ( emb('cat') * 0.6 , emb('astronaut') * 0.7 ) 
	
	# concantate embeddings by stacking their vectors
	concat( emb('mona')*0.3 , emb('lisa')*0.3 , emb('wearing'), emb('sunglasses') ) 

	# reduce embedding to 1-vector by summing all of its vectors
	reduce( emb('mdjrny-ppc') )  

	# extract vector2 from the embedding
	extract(  emb('mdjrny-ppc') , [2] ) 

	# remove vectors 3 and 5 from the embedding
	remove(  emb('mdjrny-ppc') , [3,5] ) 
	
	# evaluate and apply this eval string on the embedding
	process ( emb('cat'), '=(1*(v>=0)-1*(v<0))/50' ) 

---

The functions above can be combined to create complex formulas.
Log will show function calls, but not all operations, as the string is parsed by python itself.
You can copy the formula from the text box, and save it for later use.

Eval string usage is the same as Embedding inspector, but also following variables are available:
vec_mag: magnitude of the vector
vec_min: minimum value in the vector
vec_max: maximum value in the vector

note: vector size is 768 for SD1 and 1024 for SD2, different vector sizes can not be intermixed.
