import { Box, Button } from '@mui/material'
import { FileUpload } from '@mui/icons-material'

export default function UploadImages({ handleImageUpload, disabled, buttonText='Upload Images', image=true, video=true }) {
    return (
        <Box>
            <input
                accept={`${image ? 'image/*' : ''}${image && video ? ',' : ''}${video ? 'video/mp4' : ''}`}
                style={{ display: 'none' }}
                id={`image-upload-button-${buttonText}`}
                multiple
                type='file'
                onChange={handleImageUpload}
                disabled={disabled}
            />
            <label htmlFor={`image-upload-button-${buttonText}`}>
                <Button
                    variant='contained'
                    component='span'
                    startIcon={<FileUpload />}
                    disabled={disabled}
                    sx={{ width: '200px' }}
                >
                    {buttonText}
                </Button>
            </label>
        </Box>
    )
}