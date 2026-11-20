REGLE DU JEU DE TEST (v2)

Constat. La v1 a mesure le jeu de test une seule fois : macro-F1 0,9327 sans abstention ; avec abstention au seuil 0,90, couverture 90,4 % et precision 98,07 % sur les images traitees. Le verrou de la v1 est le fichier 05_deep_learning/results/TEST_EVALUATED.lock. La v2 reutilise les memes images de test (decoupage seed 2026 : 21 117 / 2 640 / 2 640). Ce jeu a donc deja ete consulte une fois.

Regle 1. Pour le modele de vision, tous les reglages se font sur la validation uniquement : hyperparametres, seuil d'abstention, temperature de calibration, choix du modele.

Regle 2. Le test est evalue une seule fois pour la v2, a l'etape P3.9, avec un modele et un seuil figes. Les empreintes SHA-256 du modele et du manifeste sont enregistrees avant la mesure et verifiees par le script.

Regle 3. Apres la mesure, un second verrou est pose : 05_deep_learning/results/TEST_V2_EVALUATED.lock. Le script d'evaluation de la v2 n'a pas d'option pour contourner ce verrou.

Regle 4. Tout resultat sur le test presente dans le CV, le README ou LinkedIn porte la mention : jeu de test deja consulte une fois par la v1.

Regle 5. Si un changement est decide apres la mesure du test, il est note comme tel, le test n'est pas remesure pour l'ajuster, et un nouveau decoupage documente serait necessaire pour obtenir un juge neutre.

Les evaluations du RAG et de l'agent utilisent leurs propres jeux de questions et de scenarios, avec leurs propres regles.

