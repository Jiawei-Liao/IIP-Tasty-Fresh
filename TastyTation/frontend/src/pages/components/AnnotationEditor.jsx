import { Box, Button, Typography, IconButton, Paper } from '@mui/material'
import { ArrowBack, ArrowForward, Close } from '@mui/icons-material'
import EditableAnnotatedImage from './EditableAnnotatedImage'

/**
 * Annotation editor for NewAnnotations.
 * @see VerificationAnnotationEditor.jsx for another version of AnnotationEditor for Verify
 * @param {[{annotations: [{bbox: [float], class_id: Int}], image_path: String}]} annotations: An array of annotation data objects
 * @param {Int} currentIndex: Current index of the image being annotated from annotations variable
 * @param {React.Dispatch<Int>} setCurrentIndex: Set new index for current annotation index, used to go to previous or next image
 * @param {[{id: Int, name: String}]} annotationCLasses: Array of annotation class id and name objects, optional
 * @param {function} onAnnotationsChange: Callback function to update backend when any annotations changes
 * @returns {JSX.Element} An editable annotated image with navigation functionality
 */
export default function AnnotationEditor({ annotations, currentIndex, setCurrentIndex, annotationClasses, onAnnotationsChange }) {
    // Return null if invalid
    if (!annotations.length || currentIndex < 0 || currentIndex >= annotations.length) {
        return null
    }

    const currentImage = annotations[currentIndex]

    // Navigation function to go to previous or next image
    function navigate(direction) {
        const newIndex = currentIndex + direction
        if (newIndex >= 0 && newIndex < annotations.length) {
            setCurrentIndex(newIndex)
        }
    }

    return (
        <Paper sx={{ position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh', display: 'flex', p: 2, flexDirection: 'column', borderRadius: 0 }}>
            {/* Header Section */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', p: 2, position: 'relative' }}>
                {/* Navigation Section */}
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 3, position: 'absolute', left: '50%', transform: 'translateX(-50%)' }}>
                    <Button onClick={() => navigate(-1)} disabled={currentIndex === 0} startIcon={<ArrowBack />} variant="outlined" size="large">
                        Previous
                    </Button>
                    <Typography variant="h6" color="text.primary">
                        {currentIndex + 1} of {annotations.length}
                    </Typography>
                    <Button onClick={() => navigate(1)} disabled={currentIndex === annotations.length - 1} endIcon={<ArrowForward />} variant="outlined" size="large">
                        Next
                    </Button>
                </Box>

                {/* Close Button */}
                <IconButton onClick={() => setCurrentIndex(-1)} aria-label="Close" sx={{ position: 'absolute', right: 16 }}>
                    <Close fontSize="large" />
                </IconButton>
            </Box>

            {/* Annotated Image Section */}
            <Box sx={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center', borderRadius: 1, p: 3, mt: 3, boxShadow: 1, overflow: 'hidden' }}>
                <EditableAnnotatedImage
                    item={currentImage}
                    onAnnotationsChange={onAnnotationsChange}
                    annotationClasses={annotationClasses}
                />
            </Box>
        </Paper>
    )
}
