import { useEffect, useState } from 'react'
import { Box, Paper, Stack, Typography, Button } from '@mui/material'

import SegmentObjects from './classifier components/SegmentObjects'
import CreateNewClassifier from './classifier components/CreateNewClassifer'
import Classifier from './classifier components/Classifier'

export default function Classifiers() {
    const [route, setRoute] = useState('Segment Objects')
    const [classifiers, setClassifiers] = useState([])
    const [showCreateNewClassifier, setShowCreateNewClassifier] = useState(false)

    useEffect(() => {
        // Fetch classifiers
        fetch('/api/get-classifiers')
            .then((response) => response.json())
            .then((data) => setClassifiers(data.classifiers))
            .catch((error) => console.error(error))
    }, [])
    
    function renderContent() {
        switch (route) {
            case 'Segment Objects':
                return <SegmentObjects />
            default:
                return <Classifier route={route} />
        }
    }

    return (
        <Box sx={{ display: 'flex', margin: 4, gap: 2, height: '80vh' }}>
            {/* Navigation Menu */}
            <Paper sx={{ flex: '0 0 20%', maxHeight: '100%', overflow: 'auto', display: 'flex', flexDirection: 'column' }}>
                <Box sx={{ p: 3, display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <Typography variant='h5'>Tools</Typography>
                    <Stack spacing={2}>
                        <Button variant={route === 'Segment Objects' ? 'contained' : 'outlined'} onClick={() => setRoute('Segment Objects')} sx={{ textTransform: 'none', py: 1.5 }}>
                            Segment Objects
                        </Button>
                    </Stack>
                    <Typography variant='h5'>Classifiers</Typography>
                        <Stack spacing={2}>
                            <Button variant='outlined' onClick={() => setShowCreateNewClassifier(true)} sx={{ textTransform: 'none', py: 1.5 }}>
                                Create New Classifier
                            </Button>
                            {classifiers.map((classifier, index) => (
                                <Button key={index} variant={route === classifier ? 'contained' : 'outlined'} onClick={() => setRoute(classifier)} sx={{ textTransform: 'none', py: 1.5 }}>
                                    {classifier}
                                </Button>
                            ))}
                    </Stack>
                </Box>
            </Paper>

            {/* Content */}
            <Paper sx={{ flex: '1 1 80%', maxHeight: '100%', overflow: 'auto', p: 2 }}>
                {renderContent()}
            </Paper>

            {/* Create New Classifier Modal */}
            {showCreateNewClassifier && <CreateNewClassifier setShowCreateNewClassifier={setShowCreateNewClassifier} setRoute={setRoute} setClassifiers={setClassifiers} />}
        </Box>
    )
}