-- EXEMPLE DE DEMONSTRATION du lineage SQL (n'est pas une etape du pipeline AgriLens).
CREATE TABLE class_split_summary AS
SELECT
    m.display_fr AS classe,
    m.split      AS partition,
    COUNT(*)     AS n_images
FROM split_manifest_v2 AS m
WHERE m.split IN ('train', 'val', 'test')
GROUP BY m.display_fr, m.split
