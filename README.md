# subword-mikolov
This is an implementation of English subword split method proposed in T. Mikolov (2012).  
Details that are not explicitly described in the paper are inferred based on the example input-output pair in the paper.

* Usage:

		$ python subword.py <corpus-filename> <W-parameter> <S-parameter>
		
* Example (Mikolov, et al., 2012):
		
		INPUT:	new company dreamworks interactive
		OUTPUT:	new company dre+ am+ wo+ rks: in+ te+ ra+ cti+ ve:

* Reference:

	Mikolov, T., Sutskever, I., Deoras, A., Le, H. S., Kombrink, S., & Cernocky, J. (2012b). *Subword language modeling with neural networks.* preprint. [[pdf](http://www.fit.vutbr.cz/imikolov/rnnlm/char.pdf)]
