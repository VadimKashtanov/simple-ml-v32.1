//faire un truc beaucoup plus simple.
//Pas de parametring moche. A la limite mettre des constantes modifiables lors de l'execution. (constantes dans le package)

//faire un meta fichier, oui il y a des algorithm qui generer des models.
//En fait c'est comme le Gradient Desente avec des parametres, sauf qu'un changement de parametre est une modification du models.
//Ca peut etre une grille de reseau de neurone ou peut importe.

//C'est le meta-model-trainning.
//C'est un model ou les parametres definissent le modele, et le score est la performance du trainning du model sortant.

//Le meta-model donne en sortie un modele. Le Loss est en fait juste un Train_t de ce modele avec des optimizer
//on voit ou le model va en `n` batchs, si il arrive a passer des barrieres, si il arrive a correller des donnes que les autres models n'arrivent pas ...


meta {
consts:
	float params;
funcs:
	void generate_model();
	void loss();	//tester la vitesse d'optimization moyenne, tester la mutabilite ...
}

//cree aussi des meta-optimizers : models qui cherchent des optimizer.


//En realite tout ca se fait juste avec la lib classique, puis on code un programme qui utilise ma lib,
//puis fait tout le boulo. Le `output` du meta-model serait

//Sauf que c'est juste le model, puis la recherche des models. Donc cette couche binaire model/meta-model est pultot naturelle.
//Bon cree un Lib V-0.33 ou juste un 2nd kernel.
//Ou juste un package avec ca, et les libs pour.

//	Oublier Tout ca

typedef struct scores_compute {
	//	
	Train_t * train;
	Data_t * data;

	//	Ranking
	float * scores;
	float * scores_d;
	uint * rank;
	uint * rank_d;

	//	Score methode
	uint score_id;
} Score_t;

typedef struct optimizer_model {
	//	Score
	Score_t * score;

	//	Optimizer
	uint optimizer;
	void * opti_space;	//alloc what ever the optimizer require
} Opti_t;

typedef struct genetic_set_selection {
	//	Score
	Score_t * score;

	//	Genetics
	uint genetic_methode;
	void * gtic_space;
} Gtics_t;

Score_t * score_mk(Train_t * train, Data_t * data, uint score);