import { useState } from 'react'
import { Modal, Box, Paper, Button, TextField, Typography } from '@mui/material'
import ErrorInfoSnackbar from '../components/ErrorInfoSnackbar'

export default function CreateNewClassifier({ setShowCreateNewClassifier, setRoute, setClassifiers }) {
    const [classifierName, setClassifierName] = useState('')
    const [creating, setCreating] = useState(false)
    const [error, setError] = useState('')

    function create() {
        setCreating(true)

        // Send request to create new classifier
        const formData = new FormData()
        formData.append('classifier_name', classifierName.trim())

        fetch('/api/create-classifier', {
            method: 'POST',
            body: formData
        })
            .then((response) => {
                if (!response.ok) {
                    throw new Error('Failed to create new classifier')
                } else {
                    setClassifiers((classifiers) => [...classifiers, classifierName.trim()])
                    setRoute(classifierName.trim())
                    setShowCreateNewClassifier(false)
                }
            })
            .catch((error) => {
                setError(error.message)
                setCreating(false)
            }
        )
    }

    return (
        <>
            <ErrorInfoSnackbar error={error} setError={setError} info={creating} infoMessage='Creating New Classifier...' />

            <Modal open={true} onClose={() => setShowCreateNewClassifier(false)}>
                <Paper sx={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', p: 4, display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <Typography variant='h4'>Create New Classifier</Typography>
                    <TextField label='Classifier Name' variant='outlined' value={classifierName} onChange={(e) => setClassifierName(e.target.value)} />
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', gap: 1 }}>
                        <Button variant='outlined' onClick={() => setShowCreateNewClassifier(false)}>
                            Cancel
                        </Button>
                        <Button variant='contained' onClick={create} disabled={!classifierName.trim() || creating}>
                            Create
                        </Button>
                    </Box>
                </Paper>
            </Modal>
        </>
        
    )
}