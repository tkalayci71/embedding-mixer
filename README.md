# Embedding-mixer version 0.3

Similar to Embedding Inspector, more powerful but not user-friendly, text-only, for advanced users. Not tested, provided as-is.

# Formula examples

---
	emb('cat')		# retrieves the vector of internal embedding 'cat</w>'
	emb('#2368')	# same as above, but with its ID
	emb('mdjrny-ppc')	# retrieves this loaded embedding (may contain multiple vectors)

	# mix embeddings by adding their vector values (if vector counts differ, pad with zeros)
	mix ( emb('cat') * 0.6 , emb('astronaut') * 0.7 ) 
	
	# concatenate embeddings by stacking their vectors
	concat( emb('mona')*0.3 , emb('lisa')*0.3 , emb('wearing'), emb('sunglasses') ) 

	# reduce embedding to 1-vector by summing all of its vectors
	reduce( emb('mdjrny-ppc') )  

	# extract vector2 from the embedding
	extract(  emb('mdjrny-ppc') , [2] ) 

	# remove vectors 3 and 5 from the embedding
	remove(  emb('mdjrny-ppc') , [3,5] ) 
	
	# evaluate and apply this eval string on the embedding
	process ( emb('cat'), '=(1*(v>=0)-1*(v<0))/50' ) 

	# example of a compound formula
	mix ( process(emb('cat'),'=v*(i<300)') , process(emb('dog'),'=v*(i>=300)') )

	# scale magnitude to 1
	process(  emb('cat')  , '=v/vec_mag')
	
	# keep n maximum/minimum values in vector, zero others
	keepabsmax( emb('elephant'), 100) 
	mix( keepmax( emb('elephant'), 50) , keepmin( emb('elephant'), 50) )

	# find most similar embedding to a unit vector
	mostsimilar( process( torch.zeros(768) , '=(i==300)' ) )

	# create random embedding
	torch.randn(768)/50

	# load tensor saved by inspector
	torch.tensor([[ 1.2886e-02, -7.1144e-03, -6.7101e-03, -3.5076e-03,  6.6986e-03, ......... ]])


---

The functions above can be combined to create complex formulas.
Log will show function calls, but not all operations, as the string is parsed by python itself.
You can copy the formula from the text box, and save it for later use.

Eval string usage is the same as Embedding inspector, but also following variables are available:
vec_mag: magnitude of the vector, vec_min: minimum value in the vector, vec_max: maximum value in the vector

note: vector size is 768 for SD1 and 1024 for SD2, different vector sizes can not be intermixed.

# Run script feature:


---
	#Formula
	emb(name1)*weight1+emb(name2)*weight2

---

---
	#Script
	global name1, name2, weight1, weight2
	for n in range(5):
		name1 ='chicken'
		name2 ='dinosaur'
		weight1 = n/5
		weight2 = 1-n/5
		fnam = 'test'+str(n)
		do_save('', formula_str , fnam, True)

---

log and graph may not be updated but running the above script will save 5 embeddings.