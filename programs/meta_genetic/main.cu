/*
	Il y a des allele qui generent les models. Ca fait un peuple de models.

	Les models sont en plusieurs sets et ils s'optimizent avec Train_t.
	Les meilleur sont donc selectionnes apres un algo de genetique et 
	
	Donc le score de chaque modele, est son meilleur `set`.

	On applique la selection naturelle et on fait des mutations.

	Et puis on generer de nouveau model et puis rebelotte.
*/

typedef struct {
	uint mdls;		//current models
	Mdl_t ** mdl;
} Peuple_t;