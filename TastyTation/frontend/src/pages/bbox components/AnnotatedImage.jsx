import { useRef } from 'react'
import { Box } from '@mui/material'
import BoundingBox from './BoundingBox'
import { useImageDimensions } from '../../hooks/useImageDimensions'

/**
 * Used by NewAnnotations and Verify to show an image with annotations. 
 * Shows label class names if annotationClasses is provided.
 * Passes the option to allow highlighted boxes down to BoundingBox.
 * Annotations created here are not editable.
 * @see EditableAnnotatedImage.jsx for editable version, which also updates backend.
 * @param {{annotations: [{bbox: [Float], class_id: Int}], dataset_inconsistency_index: [Int], image_path: String, inconsistencies: [String], verified_annotations: [{bbox: [Float], class_id: Int}], verified_inconsistency_index: [Int] }} item: Object of annotation data for an image when called by Verify
 * @param {{annotations: [{bbox: [Float], class_id: Int}], image_path: String}} item: Object of annotation data for an image when called by NewAnnotations
 * @param {[{id: Int, name: String}]} annotationCLasses: Array of annotation class id and name objects, optional
 * @param {Boolean} highlight: Whether to highlight labels based on inconsistency, optional
 * @returns {JSX.Element} An image with annotations with optional labels and highlights
 */
export default function AnnotatedImage({ item, annotationClasses, highlight }) {
    const imageRef = useRef(null)
    const containerRef = useRef(null)
    const imageDimensions = useImageDimensions(imageRef, containerRef)

    return (
        <Box ref={containerRef} sx={{ position: 'relative', overflow: 'hidden', height: annotationClasses? '100%' : '300px', width: '100%', backgroundColor: '#f3f4f6', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            {/* Image */}
            <img 
                ref={imageRef} 
                src={item.image_path} 
                alt={item.filename} 
                style={{ width: '100%', height: '100%', objectFit: 'contain' }} 
            />

            {/* Loop through annotations to create bounding boxes */}
            <Box sx={{ position: 'absolute', width: imageDimensions.width, height: imageDimensions.height, top: imageDimensions.offsetY, left: imageDimensions.offsetX }}>
                {item.annotations.map((annotation, index) =>
                    <BoundingBox
                        key={index}
                        bbox={annotation.bbox}
                        imageDimensions={imageDimensions}
                        className={annotationClasses && annotationClasses.find(item => item.id === annotation.class_id)?.name}
                        highlighted={highlight && item.verified_inconsistency_index?.includes(index)}
                    />
                )}
            </Box>
        </Box>
    )
}