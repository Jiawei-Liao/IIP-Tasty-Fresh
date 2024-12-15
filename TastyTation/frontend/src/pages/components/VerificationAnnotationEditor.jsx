import { Box, Button, Typography, IconButton, Paper, Tooltip } from '@mui/material'
import { ArrowBack, ArrowForward, Close, HelpOutline } from '@mui/icons-material'
import EditableAnnotatedImage from './EditableAnnotatedImage'
import AnnotatedImage from './AnnotatedImage'

export default function VerificationAnnotationEditor({ annotations, currentIndex, setCurrentIndex, annotationClasses, onAnnotationsChange, resolveInconsistencies, updateRemoveLabel }) {
    function navigate(direction) {
        const newIndex = currentIndex + direction
        if (newIndex >= 0 && newIndex < annotations.length) {
            setCurrentIndex(newIndex)
        }
    }

    if (!annotations.length || currentIndex < 0 || currentIndex >= annotations.length) {
        return null
    }

    const currentImage = annotations[currentIndex]

    return (
        <Paper sx={{ position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh', display: 'flex', p: 2, flexDirection: 'column', borderRadius: 0 }}>
            {/* Header Section */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', p: 2, position: 'relative' }}>
                {/* Resolve Inconsistencies Button */}
                <Button onClick={() => resolveInconsistencies(currentImage.image_path)} variant="contained" size="large" sx={{ backgroundColor: 'green', position: 'absolute', left: 16 }}>
                    Resolve
                </Button>
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

            {/* Annotated Image Labels Section */}
            <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2 }}>
                <Box sx={{ width: '50%', display: 'flex', justifyContent: 'center' }}>
                    <Typography variant="h6" sx={{ textAlign: 'center' }}>
                        {`Model Predicted Annotations (${currentImage.verified_annotations.length})`}
                    </Typography>
                </Box>
                <Box sx={{ width: '50%', display: 'flex', justifyContent: 'center' }}>
                    <Typography variant="h6" sx={{ textAlign: 'center' }}>
                       {`Your Annotations (${currentImage.annotations.length})`}
                    </Typography>
                </Box>
            </Box>

            {/* Issues Section */}
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', p: 1 }}>
                <Typography variant="body1">
                {currentImage.inconsistencies
                    .map(word => word.toLowerCase().replace(/^\w/, c => c.toUpperCase()))
                    .join(', ')}
                </Typography>
                <Tooltip title="Note: In some cases, overlapping labels may obscure highlights, making them difficult to distinguish. Additionally, there is a possibility that verified data may contain inaccuracies." arrow>
                    <IconButton>
                        <HelpOutline />
                    </IconButton>
                </Tooltip>
            </Box>

            {/* Annotated Image Section */}
            <Box sx={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center', borderRadius: 1, p: 3, boxShadow: 1, overflow: 'hidden' }}>
                <Box sx={{ display: 'flex', width: '100%', height: '100%', gap: 2 }}>
                    <Box sx={{ width: '50%' }}>
                        <AnnotatedImage
                            item={{
                                ...currentImage,
                                annotations: currentImage.verified_annotations
                            }}
                            annotationClasses={annotationClasses}
                            highlight={true}
                        />
                    </Box>
                    <Box sx={{ width: '50%' }}>
                        <EditableAnnotatedImage
                            item={currentImage}
                            onAnnotationsChange={onAnnotationsChange}
                            newAnnotationClasses={annotationClasses}
                            highlight={true}
                            updateRemoveLabel={updateRemoveLabel}
                        />
                    </Box>
                </Box>
            </Box>
        </Paper>
    )
}
